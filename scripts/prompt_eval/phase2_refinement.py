"""Phase 2: refine top cells per interpreter across temperature ∈ {0.0, 0.2, 0.4}.

For each interpreter, find the top 2 (model, variant) cells in
phase 1b, then sweep temperature ∈ {0.0, 0.2, 0.4} on the same
intents. Each temperature writes to a separate output JSONL so the
harness's resume key set stays coherent.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .harness import HarnessConfig, run_harness
from .intents_ast_select import AST_SELECT_INTENTS
from .intents_plucker import PLUCKER_INTENTS
from .intents_pss import PSS_INTENTS
from .intents_python import PYTHON_INTENTS


def _top_cells_from_grid(
    grid_jsonl: Path, top_n: int = 2
) -> dict[str, list[tuple[str, str]]]:
    """Rank (model, variant) cells per interpreter by total score."""
    buckets: dict[str, dict[tuple[str, str], dict]] = defaultdict(
        lambda: defaultdict(lambda: {"score": 0, "rows": 0, "latencies": []})
    )
    with grid_jsonl.open() as f:
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
            interp = row.get("interpreter", "?")
            key = (row.get("model", "?"), row.get("variant_id", "?"))
            b = buckets[interp][key]
            b["score"] += int(row.get("score", 0))
            b["rows"] += 1
            b["latencies"].append(float(row.get("duration_ms_generation", 0.0)))

    out: dict[str, list[tuple[str, str]]] = {}
    for interp, keyed in buckets.items():
        ranked = sorted(
            (
                (k, v["score"],
                 statistics.median(v["latencies"]) if v["latencies"] else 0.0)
                for k, v in keyed.items()
            ),
            key=lambda t: (-t[1], t[2]),
        )
        out[interp] = [k for k, _s, _l in ranked[:top_n]]
    return out


_ALL_INTENTS = {
    "python": PYTHON_INTENTS,
    "ast-select": AST_SELECT_INTENTS,
    "pss": PSS_INTENTS,
    "plucker": PLUCKER_INTENTS,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("results/prompt-eval-2026-04-11/phase1b-grid.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/prompt-eval-2026-04-11/phase2-refinement.jsonl"),
    )
    parser.add_argument(
        "--toybox",
        type=Path,
        default=Path("tests/eval/fixtures/toybox"),
    )
    args = parser.parse_args()

    if not args.grid.exists():
        raise SystemExit(
            f"grid JSONL not found: {args.grid}. Run phase1b_grid first."
        )
    top = _top_cells_from_grid(args.grid, top_n=2)

    for temperature in (0.0, 0.2, 0.4):
        temp_suffix = f".t{temperature}"
        temp_output = args.output.with_name(
            args.output.stem + temp_suffix + args.output.suffix
        )
        for interpreter, cells in top.items():
            if not cells:
                continue
            models = sorted({m for m, _v in cells})
            variant_ids = sorted({v for _m, v in cells})
            intents = _ALL_INTENTS[interpreter]
            cfg = HarnessConfig(
                output_path=temp_output,
                models=models,
                interpreters=[interpreter],
                variant_ids=variant_ids,
                intents=intents,
                toybox_dir=args.toybox,
                ollama_host=args.host,
                temperature=temperature,
                timeout=args.timeout,
            )
            print(
                f"\n== phase 2 :: t={temperature} "
                f"interpreter={interpreter} =="
            )
            print(f"models={models}  variants={variant_ids}  -> {temp_output}")
            run_harness(cfg)


if __name__ == "__main__":
    main()
