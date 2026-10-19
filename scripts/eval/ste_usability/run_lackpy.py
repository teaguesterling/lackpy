"""Generate the eval programs through lackpy's own generation path.

usage: run_lackpy.py RUN_DIR LABEL ROUTE
  ROUTE: an explicit device model route, such as
         tiiny-fast/Qwen/Qwen3-30B-A3B-Instruct. It goes to woollama as
         "openai/<ROUTE>" with base_url http://127.0.0.1:47600/v1 (the device
         gateway is OpenAI-compatible). Override the gateway with LACKPY_EVAL_BASE_URL.
  For each task: copy fixtures/ to a fresh directory, then call
  LackpyService(workspace=<copy>).generate(goal, model="openai/<ROUTE>").
  -> RUN_DIR/LABEL/r1/<id>.py         the program lackpy returned (valid by construction)
     RUN_DIR/LABEL/r1/<id>.meta.json  secs, provider, correction strategy and attempts
     RUN_DIR/LABEL/r1/<id>.error.txt  when generation failed; no .py is written, and
                                      score.py counts the missing program as a failure
THINKING=off adds chat_template_kwargs.enable_thinking=false (as run_local.py does).

Differences from run_local.py, which this mirrors:
  - The prompt is lackpy's build_system_prompt (system message) and the goal
    (user message), not prompts.task_prompt; it holds no workspace file contents.
  - lackpy runs its own cleanup and correction chain, so a program that needed a
    repair is in r1, marked by correction_strategy in its meta.json. There is no r2.
  - woollama does not stream. The gateway's 120 s idle timeout can end a long
    generation; that shows as an error.txt.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tasks import TASKS  # noqa: E402

from lackpy.config import LackpyConfig  # noqa: E402
from lackpy.service import LackpyService  # noqa: E402

BASE_URL = os.environ.get("LACKPY_EVAL_BASE_URL", "http://127.0.0.1:47600/v1")
USAGE = "usage: run_lackpy.py RUN_DIR LABEL ROUTE"


def parse_args(argv: list[str]) -> tuple[Path, str, str]:
    if len(argv) != 3:
        raise SystemExit(USAGE)
    run_dir, label, route = argv
    if "/" not in route or route.startswith("openai/"):
        raise SystemExit(f"{USAGE}\nROUTE must be an explicit device route such as "
                         f"tiiny-fast/<model id>, without a provider prefix; got {route!r}")
    return Path(run_dir), label, route


def model_for(route: str) -> str:
    return f"openai/{route}"


def make_config(workspace: Path, route: str) -> LackpyConfig:
    params: dict = {"max_tokens": 4096}
    if os.environ.get("THINKING") == "off":
        params["chat_template_kwargs"] = {"enable_thinking": False}
    return LackpyConfig(
        inference_order=["woollama"],
        inference_providers={"woollama": {
            "plugin": "woollama", "model": model_for(route), "temperature": 0.2,
            "api_key": "local", "base_url": BASE_URL, "params": params,
        }},
        config_dir=workspace / ".lackpy",
    )


def generate_one(task: dict, route: str, out: Path, service_factory=None) -> bool:
    """Generate one task's program into OUT. Return whether a program was written."""
    factory = service_factory or (lambda ws: LackpyService(workspace=ws, config=make_config(ws, route)))
    with tempfile.TemporaryDirectory(prefix="lackpy-eval-") as tmp:
        ws = Path(tmp) / "ws"
        shutil.copytree(HERE / "fixtures", ws)
        service = factory(ws)
        t0 = time.time()
        try:
            gen = asyncio.run(service.generate(task["goal"], model=model_for(route)))
        except Exception as e:  # a failed generation is a result, never a skip
            secs = time.time() - t0
            (out / f"{task['id']}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            (out / f"{task['id']}.meta.json").write_text(json.dumps({"secs": round(secs, 1), "ok": False}))
            return False
        secs = time.time() - t0
    (out / f"{task['id']}.py").write_text(gen.program.rstrip() + "\n")
    (out / f"{task['id']}.meta.json").write_text(json.dumps({
        "secs": round(secs, 1), "ok": True, "provider": gen.provider_name,
        "correction_strategy": gen.correction_strategy,
        "correction_attempts": gen.correction_attempts,
    }))
    return True


def main(argv: list[str] | None = None, tasks: list[dict] | None = None, service_factory=None) -> None:
    run_dir, label, route = parse_args(sys.argv[1:] if argv is None else argv)
    out = run_dir / label / "r1"
    out.mkdir(parents=True, exist_ok=True)
    for t in TASKS if tasks is None else tasks:
        t0 = time.time()
        ok = generate_one(t, route, out, service_factory)
        print(f"{t['id']} {'ok' if ok else 'FAILED'} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
