"""Quick A/B trial runner for ad-hoc tasks.

Runs a single natural-language task through N configurations
(model × variant), scores each, and returns a ranked comparison.
Appends every trial to a persistent log for the advisor to learn from.

Usage:
    python -m scripts.prompt_eval.trial "find callers of validate_token" \
        --models qwen2.5:1.5b,qwen2.5-coder:3b \
        --variants baseline,specialized \
        --interpreter python
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .eval_kit import build_eval_kit
from .intents import GateResult, Intent
from .prompts import get_prompt
from .runner import GenerationRecord, generate_once, make_ollama_client
from .scoring import CellScore, run_gate, score_cell


# ── Gate factories per interpreter ─────────────────────────────────────
# These are simplified gates for ad-hoc tasks (no intent-specific assertions).

def _python_adhoc_gate(program: str) -> GateResult:
    from lackpy.lang.validator import validate
    result = validate(program, allowed_names={"read_file", "find_files", "find_def", "find_refs"})
    return GateResult(passed=result.valid, errors=list(result.errors))


def _selector_adhoc_gate(program: str) -> GateResult:
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty"])
    s = program.strip()
    if "{" in s or "}" in s:
        return GateResult(passed=False, errors=["braces in bare selector"])
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) > 1:
        return GateResult(passed=False, errors=["multiline"])
    if not s.startswith((".", "#", "*", "[")):
        return GateResult(passed=False, errors=["invalid selector start"])
    return GateResult(passed=True)


def _pss_adhoc_gate(program: str) -> GateResult:
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty"])
    if program.count("{") != program.count("}"):
        return GateResult(passed=False, errors=["unbalanced braces"])
    return GateResult(passed=True)


def _plucker_adhoc_gate(program: str) -> GateResult:
    import ast
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty"])
    try:
        tree = ast.parse(program.strip(), mode="eval")
    except SyntaxError as e:
        return GateResult(passed=False, errors=[f"parse: {e}"])
    current = tree.body
    has_chain = False
    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                has_chain = True
                current = current.func.value
                continue
            if isinstance(current.func, ast.Name):
                if current.func.id != "source":
                    return GateResult(passed=False, errors=[f"root is {current.func.id}, not source"])
                break
            return GateResult(passed=False, errors=["root not a Name call"])
        if isinstance(current, ast.Attribute):
            has_chain = True
            current = current.value
            continue
        return GateResult(passed=False, errors=["unexpected node"])
    if not has_chain:
        return GateResult(passed=False, errors=["bare source()"])
    return GateResult(passed=True)


_ADHOC_GATES = {
    "python": _python_adhoc_gate,
    "ast-select": _selector_adhoc_gate,
    "pss": _pss_adhoc_gate,
    "plucker": _plucker_adhoc_gate,
}


# ── Trial result ───────────────────────────────────────────────────────


@dataclass
class TrialResult:
    """Result of one configuration on one ad-hoc task."""

    model: str
    variant: str
    interpreter: str
    task: str
    raw_generation: str
    sanitized_program: str
    gate_passed: bool
    gate_errors: list[str]
    executed: bool
    exec_output: Any
    exec_output_repr: str
    exec_error: str | None
    duration_ms: float
    tokens_eval: int
    tokens_prompt: int
    score: int  # 0=gate fail, 1=exec fail, 2=exec success


# ── Trial runner ───────────────────────────────────────────────────────


def _namespace_desc_for(interpreter: str, toybox_dir: Path) -> str:
    if interpreter == "python":
        kit = build_eval_kit(toybox_dir)
        return kit.description
    return f"(interpreter={interpreter}; see prompt variant for syntax reference)"


def quick_trial(
    task: str,
    interpreter: str = "python",
    models: list[str] | None = None,
    variants: list[str] | None = None,
    toybox_dir: Path = Path("tests/eval/fixtures/toybox"),
    ollama_host: str = "http://localhost:11435",
    temperature: float = 0.2,
    timeout: int = 60,
    log_path: Path = Path("results/trials.jsonl"),
) -> list[TrialResult]:
    """Run a task through multiple configurations and return ranked results.

    Each configuration = (model, variant). Results are ranked by score
    (desc), then latency (asc). Every trial is appended to `log_path`
    for the advisor to learn from.
    """
    if models is None:
        models = ["qwen2.5:1.5b", "qwen2.5-coder:3b", "qwen2.5:3b"]
    if variants is None:
        variants = ["baseline", "specialized"]

    toybox_dir = Path(toybox_dir).resolve()
    gate_fn = _ADHOC_GATES[interpreter]
    namespace_desc = _namespace_desc_for(interpreter, toybox_dir)

    # Build an ad-hoc Intent for scoring (assertion always passes since we don't have ground truth)
    adhoc_intent = Intent(
        id="adhoc",
        interpreter=interpreter,
        difficulty="adhoc",
        text=task,
        return_shape="unknown",
        structural_gate=gate_fn,
        exec_assertion=lambda x: x is not None,
    )

    client = make_ollama_client(host=ollama_host)
    results: list[TrialResult] = []

    for model in models:
        for variant in variants:
            system_prompt = get_prompt(interpreter, variant, namespace_desc)
            gen = generate_once(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_message=task,
                temperature=temperature,
                timeout=timeout,
            )

            if gen.error or not gen.raw:
                results.append(TrialResult(
                    model=model, variant=variant, interpreter=interpreter,
                    task=task, raw_generation=gen.raw,
                    sanitized_program="", gate_passed=False,
                    gate_errors=[gen.error or "empty generation"],
                    executed=False, exec_output=None, exec_output_repr="",
                    exec_error=gen.error, duration_ms=gen.duration_ms,
                    tokens_eval=gen.tokens_eval, tokens_prompt=gen.tokens_prompt,
                    score=0,
                ))
                continue

            cs = score_cell(adhoc_intent, gen.raw, toybox_dir)
            results.append(TrialResult(
                model=model, variant=variant, interpreter=interpreter,
                task=task, raw_generation=gen.raw,
                sanitized_program=cs.sanitized_program,
                gate_passed=cs.gate.passed,
                gate_errors=cs.gate.errors,
                executed=cs.executed,
                exec_output=cs.exec_output,
                exec_output_repr=repr(cs.exec_output)[:500] if cs.executed else "",
                exec_error=cs.exec_error,
                duration_ms=gen.duration_ms,
                tokens_eval=gen.tokens_eval,
                tokens_prompt=gen.tokens_prompt,
                score=cs.score,
            ))

    # Sort: score desc, latency asc
    results.sort(key=lambda r: (-r.score, r.duration_ms))

    # Log to persistent trial file
    _log_trials(results, log_path)

    return results


def _log_trials(results: list[TrialResult], log_path: Path) -> None:
    """Append trial results to a persistent JSONL log."""
    import datetime
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log_path.open("a") as f:
        for r in results:
            row = {
                "timestamp": ts,
                "model": r.model,
                "variant": r.variant,
                "interpreter": r.interpreter,
                "task": r.task,
                "score": r.score,
                "gate_passed": r.gate_passed,
                "executed": r.executed,
                "duration_ms": r.duration_ms,
                "tokens_eval": r.tokens_eval,
                "sanitized_program": r.sanitized_program[:200],
                "exec_output_repr": r.exec_output_repr[:200],
                "exec_error": r.exec_error,
            }
            f.write(json.dumps(row) + "\n")


# ── Display ────────────────────────────────────────────────────────────


def print_trial_results(results: list[TrialResult]) -> None:
    """Print a ranked comparison table."""
    print(f"\n{'=' * 80}")
    if results:
        print(f"TRIAL: {results[0].task[:70]}")
    print(f"{'=' * 80}")
    print(f"  {'#':<3} {'Score':>5} {'Model':<28} {'Variant':<25} {'ms':>6} {'Gate':>4} {'Exec':>4}")
    print(f"  {'─' * 75}")
    for i, r in enumerate(results, 1):
        g = "✓" if r.gate_passed else "✗"
        x = "✓" if r.executed and not r.exec_error else ("✗" if r.executed else "–")
        print(f"  {i:<3} {r.score:>5} {r.model:<28} {r.variant:<25} {r.duration_ms:>6.0f} {g:>4} {x:>4}")

    # Show top result details
    if results and results[0].score > 0:
        best = results[0]
        print(f"\n── Best: {best.model} / {best.variant} (score={best.score}) ──")
        prog = best.sanitized_program.replace("\n", "\n    ")
        print(f"  program:\n    {prog}")
        if best.exec_output_repr:
            print(f"  output: {best.exec_output_repr[:200]}")

    # Show worst failures for diagnosis
    failures = [r for r in results if r.score == 0]
    if failures:
        print(f"\n── Failures ({len(failures)}) ──")
        for r in failures[:3]:
            reason = r.gate_errors[0] if r.gate_errors else (r.exec_error or "unknown")
            prog = r.sanitized_program[:60].replace("\n", " | ")
            print(f"  {r.model} / {r.variant}: {reason[:60]}")
            if prog:
                print(f"    prog: {prog}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick A/B trial for ad-hoc tasks")
    parser.add_argument("task", help="Natural language task to trial")
    parser.add_argument("--interpreter", default="python",
                        choices=["python", "ast-select", "pss", "plucker"])
    parser.add_argument("--models", default="qwen2.5:1.5b,qwen2.5-coder:3b,qwen2.5:3b",
                        help="Comma-separated model list")
    parser.add_argument("--variants", default="baseline,specialized",
                        help="Comma-separated variant list")
    parser.add_argument("--toybox", type=Path, default=Path("tests/eval/fixtures/toybox"))
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log", type=Path, default=Path("results/trials.jsonl"))
    args = parser.parse_args()

    results = quick_trial(
        task=args.task,
        interpreter=args.interpreter,
        models=args.models.split(","),
        variants=args.variants.split(","),
        toybox_dir=args.toybox,
        ollama_host=args.host,
        temperature=args.temperature,
        timeout=args.timeout,
        log_path=args.log,
    )
    print_trial_results(results)


if __name__ == "__main__":
    main()
