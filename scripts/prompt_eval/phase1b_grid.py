"""Phase 1b: full grid over qualifier winners × 4 interpreters × 4 prompts × 14 intents."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cohort import pick_phase1b_cohort
from .harness import HarnessConfig, run_harness
from .intents_ast_select import AST_SELECT_INTENTS
from .intents_plucker import PLUCKER_INTENTS
from .intents_pss import PSS_INTENTS
from .intents_python import PYTHON_INTENTS
from .prompts import list_variant_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--qualifier",
        type=Path,
        default=Path("results/prompt-eval-2026-04-11/phase1a-qualifier.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/prompt-eval-2026-04-11/phase1b-grid.jsonl"),
    )
    parser.add_argument(
        "--toybox",
        type=Path,
        default=Path("tests/eval/fixtures/toybox"),
    )
    args = parser.parse_args()

    if not args.qualifier.exists():
        raise SystemExit(
            f"qualifier JSONL not found: {args.qualifier}. "
            "Run phase1a_qualifier first."
        )
    cohort = pick_phase1b_cohort(args.qualifier, top_n=6, gate_floor=0.5)
    # Always include qwen2.5-coder:7b — our prior top python performer.
    # It narrowly misses the auto-selection due to latency tiebreak but
    # is the strongest model with the right prompt variant.
    if "qwen2.5-coder:7b" not in cohort:
        cohort.append("qwen2.5-coder:7b")
    if not cohort:
        raise SystemExit("no models passed the qualifier floor.")
    print(f"Phase 1b cohort: {cohort}")

    all_intents = PYTHON_INTENTS + AST_SELECT_INTENTS + PSS_INTENTS + PLUCKER_INTENTS
    cfg = HarnessConfig(
        output_path=args.output,
        models=cohort,
        interpreters=["python", "ast-select", "pss", "plucker"],
        variant_ids=list_variant_ids(),
        intents=all_intents,
        toybox_dir=args.toybox,
        ollama_host=args.host,
        temperature=0.2,
        timeout=args.timeout,
    )
    run_harness(cfg)


if __name__ == "__main__":
    main()
