"""Generate programs with the local Tiiny model through `ttt code`.

usage: run_local.py RUN_DIR MODEL_LABEL {r1|r2} [ROUTE]
  ROUTE: a woollama model route such as tiiny-fast/Qwen/Qwen3.6-35B-A3B. Without
  it, `ttt code` uses tiiny-fast/default, and the device does not report which
  model that is.
  r1: answer every prompt in RUN_DIR/prompts/ -> RUN_DIR/<label>/r1/<id>.py
  r2: for every r1 program that fails validation, send a repair prompt
      -> RUN_DIR/<label>/r2/<id>.py (valid r1 programs are copied unchanged)
Each raw reply is kept beside its program as <id>.reply.txt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import extract, repair_prompt, task_prompt  # noqa: E402
from tasks import TASKS  # noqa: E402

from lackpy.lang import validate  # noqa: E402


def ask(prompt: str, route: str | None = None) -> tuple[str, float]:
    t0 = time.time()
    if route:
        return _ask_route(prompt, route), time.time() - t0
    r = subprocess.run(["ttt", "code", "-"], input=prompt, capture_output=True,
                       text=True, timeout=600)
    if r.returncode != 0:
        return f"[ttt failed rc={r.returncode}] {r.stderr[-300:]}", time.time() - t0
    return r.stdout, time.time() - t0


LAST_META: dict = {}


def _ask_route(prompt: str, route: str) -> str:
    """Call woollama with an explicit model route, streaming.

    The device gateway enforces a 120 s idle timeout, so a non-streamed request
    504s on any generation longer than that. Streaming keeps tokens flowing.
    """
    import json
    import urllib.request
    body = {"model": route, "stream": True,
            # THINKING=off asks a hybrid model (Qwen3.6) for its instruct mode.
            # Without it, Qwen3.6 spent the whole token budget reasoning and
            # returned empty content (measured 2026-10-18).
            **({"chat_template_kwargs": {"enable_thinking": False}}
               if os.environ.get("THINKING") == "off" else {}),
            "max_tokens": max(2048, min(4096, 31000 - len(prompt) // 4)),
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:47600/v1/chat/completions", method="POST",
                                 headers={"Authorization": "Bearer local",
                                          "Content-Type": "application/json"},
                                 data=json.dumps(body).encode())
    last = ""
    for attempt in range(1, 4):  # transport failures are retried, model output never is
        parts, finish = [], None
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    choices = json.loads(line[5:]).get("choices") or []
                    if choices:  # usage-only chunks carry no choices
                        parts.append((choices[0].get("delta") or {}).get("content") or "")
                        finish = choices[0].get("finish_reason") or finish
        except Exception as e:
            last = f"[route {route} failed] {e}"
            time.sleep(10 * attempt)
            continue
        if finish is None:  # the stream stopped without saying why: transport, not model
            last = f"[route {route} failed] stream ended without finish_reason"
            time.sleep(10 * attempt)
            continue
        LAST_META.update(finish_reason=finish, attempts=attempt)
        return "".join(parts)
    LAST_META.update(finish_reason=None, attempts=3)
    return last


_SEEN: set[str] = set()


def _foreign_launches(route: str | None) -> list[str]:
    """Device-log launches of a chat model other than ROUTE's, not seen before.

    The Tiiny device is shared: any client can load a model, which evicts the
    one this run is using. A run whose program files sit beside a SWAP.txt is
    not attributable to one model.
    """
    want = route.split("/", 1)[1] if route else None
    try:
        log = subprocess.run(["ttt", "logs", "-n", "120"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception as e:
        return [f"device log unavailable: {e}"]
    out = []
    for line in log.splitlines():
        if "action=launch.dispatch" not in line or "Embedding" in line or "Reranker" in line:
            continue
        if line in _SEEN:
            continue
        _SEEN.add(line)
        if want is None or f"model_id={want} " not in line:
            out.append(line)
    return out


def main() -> None:
    run_dir, label, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    route = sys.argv[4] if len(sys.argv) > 4 else None
    _foreign_launches(route)  # mark every launch already in the log as seen
    out = run_dir / label / phase
    out.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        if phase == "r1":
            reply, secs = ask(task_prompt(t), route)
        else:
            prev = (run_dir / label / "r1" / f"{t['id']}.py").read_text()
            v = validate(prev)
            if v.valid:
                (out / f"{t['id']}.py").write_text(prev)
                continue
            msgs = [x.message + (f" (hint: {x.hint})" if x.hint else "") for x in v.violations]
            reply, secs = ask(repair_prompt(t, prev, msgs), route)
        swaps = _foreign_launches(route)
        if swaps:  # another model was launched on the shared device during this call
            (out / f"{t['id']}.SWAP.txt").write_text("\n".join(swaps))
        (out / f"{t['id']}.reply.txt").write_text(reply)
        (out / f"{t['id']}.meta.json").write_text(json.dumps({"secs": round(secs, 1), **LAST_META}))
        LAST_META.clear()
        (out / f"{t['id']}.py").write_text(extract(reply))
        print(f"{t['id']} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
