"""Harness orchestrator for the prompt evaluation sweep.

Iterates the matrix {model × interpreter × variant × intent}, runs
the Ollama generation via runner.generate_once, scores via
scoring.score_cell, and writes one JSONL row per cell to a resumable
output file.

Matrix iteration order: intents fastest, variants, interpreters,
models slowest. This keeps the model hot across a whole block of
(interpreter × variant × intent) cells before switching.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lackpy.kit.registry import ResolvedKit

from .eval_kit import build_eval_kit
from .intents import Intent
from .prompts import get_prompt
from .runner import GenerationRecord, generate_once, make_ollama_client
from .scoring import CellScore, score_cell


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class HarnessConfig:
    """All inputs for a harness run."""

    output_path: Path
    models: list[str]
    interpreters: list[str]
    variant_ids: list[str]
    intents: list[Intent]
    toybox_dir: Path
    ollama_host: str = "http://localhost:11435"
    temperature: float = 0.2
    timeout: int = 60
    keep_alive: str = "30m"


# ── Hashing and namespace descriptions ─────────────────────────────────


def toybox_hash(toybox_dir: Path) -> str:
    """Return sha256 of the concatenated sorted python file contents under toybox_dir."""
    h = hashlib.sha256()
    for p in sorted(Path(toybox_dir).rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


_NAMESPACE_DESC_CACHE: dict[str, str] = {}


def _namespace_desc_for(interpreter: str, toybox_dir: Path) -> str:
    """Namespace description passed into the prompt template.

    For `python`, resolve the eval kit and use its description. For
    the three pluckit-backed interpreters, return a short placeholder
    — the prompt variants hard-code their own selector/sheet/chain
    documentation and don't need a kit rendering.
    """
    cache_key = f"{interpreter}:{toybox_dir}"
    if cache_key in _NAMESPACE_DESC_CACHE:
        return _NAMESPACE_DESC_CACHE[cache_key]
    if interpreter == "python":
        kit: ResolvedKit = build_eval_kit(toybox_dir)
        desc = kit.description
    else:
        desc = f"(interpreter={interpreter}; see prompt variant for syntax reference)"
    _NAMESPACE_DESC_CACHE[cache_key] = desc
    return desc


# ── Cell computation and resume ────────────────────────────────────────


def compute_cells(
    models: list[str],
    interpreters: list[str],
    variant_ids: list[str],
    intents: list[Intent],
) -> list[tuple[str, str, str, Intent]]:
    """Expand the matrix into an ordered list of cells.

    Intents are filtered to match each interpreter — an intent tagged
    'python' is skipped when the current interpreter is 'ast-select'.
    Ordering: model (slowest) → interpreter → variant → intent
    (fastest). This keeps a model hot across (variant × intent) cells.
    """
    cells: list[tuple[str, str, str, Intent]] = []
    intents_by_interpreter: dict[str, list[Intent]] = {}
    for intent in intents:
        intents_by_interpreter.setdefault(intent.interpreter, []).append(intent)
    for model in models:
        for interp in interpreters:
            for variant_id in variant_ids:
                for intent in intents_by_interpreter.get(interp, []):
                    cells.append((model, interp, variant_id, intent))
    return cells


def load_completed_keys(
    output_path: Path,
) -> set[tuple[str, str, str, str]]:
    """Read an existing JSONL and return the set of completed cell keys.

    The header `_meta` row is ignored. Rows missing any of the four
    key fields are also ignored so partial or corrupt files can be
    recovered gracefully.
    """
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                continue
            try:
                keys.add((
                    row["model"], row["interpreter"],
                    row["variant_id"], row["intent_id"],
                ))
            except KeyError:
                continue
    return keys


# ── Row serialization ──────────────────────────────────────────────────


def make_row(
    *,
    model: str,
    interpreter: str,
    variant_id: str,
    intent: Intent,
    gen: GenerationRecord,
    score: CellScore | None,
) -> dict[str, Any]:
    """Compose one JSONL row. Safe for both successful and error cells."""
    row: dict[str, Any] = {
        "model": model,
        "interpreter": interpreter,
        "variant_id": variant_id,
        "intent_id": intent.id,
        "intent_difficulty": intent.difficulty,
        "intent_text": intent.text,
        "return_shape": intent.return_shape,
        "raw_generation": gen.raw,
        "duration_ms_generation": gen.duration_ms,
        "tokens_eval": gen.tokens_eval,
        "tokens_prompt": gen.tokens_prompt,
        "gen_error": gen.error,
    }
    if score is not None:
        row.update({
            "sanitized_program": score.sanitized_program,
            "gate_passed": score.gate.passed,
            "gate_errors": score.gate.errors,
            "executed": score.executed,
            "exec_output_repr": repr(score.exec_output) if score.executed else None,
            "exec_error": score.exec_error,
            "assertion_passed": score.assertion_passed,
            "duration_ms_execution": score.duration_ms_execution,
            "score": score.score,
        })
    else:
        row.update({
            "sanitized_program": "",
            "gate_passed": False,
            "gate_errors": [],
            "executed": False,
            "exec_output_repr": None,
            "exec_error": None,
            "assertion_passed": False,
            "duration_ms_execution": 0.0,
            "score": 0,
        })
    return row


def _read_meta_hash(output_path: Path) -> str | None:
    """Read the first line of a JSONL and return _meta.toybox_hash if present."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None
    with output_path.open() as f:
        first_line = f.readline().strip()
    if not first_line:
        return None
    try:
        row = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    meta = row.get("_meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("toybox_hash")


def _write_meta(output_path: Path, cfg: HarnessConfig) -> None:
    """Write the _meta header row if the file doesn't already exist."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": {
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "toybox_hash": toybox_hash(cfg.toybox_dir),
            "toybox_dir": str(cfg.toybox_dir),
            "ollama_host": cfg.ollama_host,
            "models": cfg.models,
            "interpreters": cfg.interpreters,
            "variant_ids": cfg.variant_ids,
            "intent_ids": [i.id for i in cfg.intents],
            "temperature": cfg.temperature,
            "timeout": cfg.timeout,
        }
    }
    with output_path.open("w") as f:
        f.write(json.dumps(meta) + "\n")


# ── Main loop ──────────────────────────────────────────────────────────


_interrupt_received = False


def _install_sigint_handler() -> None:
    def _handler(_signum, _frame):
        global _interrupt_received
        _interrupt_received = True
        print("\n[harness] SIGINT received — finishing current cell then exiting cleanly.", file=sys.stderr)
    signal.signal(signal.SIGINT, _handler)


def run_harness(cfg: HarnessConfig) -> None:
    """Execute the matrix, writing JSONL as it goes.

    Already-completed cells (matching model, interpreter, variant_id,
    intent_id) are skipped. A KeyboardInterrupt / SIGINT causes the
    harness to finish the current cell and exit 0.
    """
    global _interrupt_received
    _interrupt_received = False
    _install_sigint_handler()

    current_hash = toybox_hash(cfg.toybox_dir)
    if cfg.output_path.exists() and cfg.output_path.stat().st_size > 0:
        stored = _read_meta_hash(cfg.output_path)
        if stored is None:
            raise RuntimeError(
                "refusing to resume: existing JSONL has no toybox_hash in _meta"
            )
        if stored != current_hash:
            raise RuntimeError(
                f"refusing to resume: toybox hash mismatch. "
                f"Stored={stored[:16]}… Current={current_hash[:16]}… "
                f"Delete the output file or revert the toybox to resume."
            )

    _write_meta(cfg.output_path, cfg)
    completed = load_completed_keys(cfg.output_path)

    cells = compute_cells(
        models=cfg.models,
        interpreters=cfg.interpreters,
        variant_ids=cfg.variant_ids,
        intents=cfg.intents,
    )
    pending = [
        c for c in cells
        if (c[0], c[1], c[2], c[3].id) not in completed
    ]

    total = len(cells)
    done_count = total - len(pending)
    print(f"[harness] {done_count}/{total} cells already completed; {len(pending)} pending.")

    client = make_ollama_client(host=cfg.ollama_host)

    with cfg.output_path.open("a") as out_f:
        with tqdm(total=total, initial=done_count, desc="eval", unit="cell") as bar:
            for model, interp, variant_id, intent in pending:
                if _interrupt_received:
                    print("[harness] interrupted — exiting.", file=sys.stderr)
                    break
                bar.set_description(
                    f"{model[:20]} / {interp[:10]} / {variant_id[:15]} / {intent.id}"
                )
                namespace_desc = _namespace_desc_for(interp, cfg.toybox_dir)
                system_prompt = get_prompt(interp, variant_id, namespace_desc)
                gen = generate_once(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_message=intent.text,
                    temperature=cfg.temperature,
                    timeout=cfg.timeout,
                    keep_alive=cfg.keep_alive,
                )
                score: CellScore | None = None
                if gen.error is None and gen.raw:
                    try:
                        score = score_cell(intent, gen.raw, cfg.toybox_dir)
                    except Exception as e:
                        score = None
                        print(
                            f"[harness] score_cell raised on {intent.id}: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                row = make_row(
                    model=model, interpreter=interp, variant_id=variant_id,
                    intent=intent, gen=gen, score=score,
                )
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()
                bar.update(1)

    print(f"[harness] run complete — {cfg.output_path}")
