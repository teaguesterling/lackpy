"""Phase 1a: qualify models on the python interpreter's core intents.

Usage:
    python -m scripts.prompt_eval.phase1a_qualifier \\
        [--host http://localhost:11435] [--timeout 60] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cohort import PHASE1A_MODELS, pick_phase1b_cohort
from .harness import HarnessConfig, run_harness
from .intents_python import PYTHON_INTENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/prompt-eval-2026-04-11/phase1a-qualifier.jsonl"),
    )
    parser.add_argument(
        "--toybox",
        type=Path,
        default=Path("tests/eval/fixtures/toybox"),
    )
    args = parser.parse_args()

    core_intents = [i for i in PYTHON_INTENTS if i.difficulty == "core"]
    cfg = HarnessConfig(
        output_path=args.output,
        models=PHASE1A_MODELS,
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=core_intents,
        toybox_dir=args.toybox,
        ollama_host=args.host,
        temperature=0.2,
        timeout=args.timeout,
    )
    run_harness(cfg)

    print("\n== Qualifier selection ==")
    cohort = pick_phase1b_cohort(args.output, top_n=6, gate_floor=0.5)
    print("Top 6 for phase 1b:")
    for m in cohort:
        print(f"  {m}")


if __name__ == "__main__":
    main()
