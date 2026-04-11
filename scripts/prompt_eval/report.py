"""JSONL → markdown report consolidator.

Usage:
    python -m scripts.prompt_eval.report results/prompt-eval-2026-04-11/*.jsonl \\
        --out results/prompt-eval-2026-04-11/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportData:
    rows: list[dict] = field(default_factory=list)
    toybox_hash: str = ""
    sources: list[str] = field(default_factory=list)


def consolidate_jsonls(paths: list[Path]) -> ReportData:
    """Merge multiple JSONL files into a single ReportData.

    Fails loudly if the files disagree on the toybox hash — cross-hash
    comparisons are invalid because the fixture drove different
    answers.
    """
    data = ReportData()
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        data.sources.append(str(p))
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in row:
                    h = row["_meta"].get("toybox_hash", "")
                    if data.toybox_hash and h and h != data.toybox_hash:
                        raise ValueError(
                            f"toybox hash mismatch across sources: "
                            f"{data.toybox_hash} != {h}"
                        )
                    if h and not data.toybox_hash:
                        data.toybox_hash = h
                    continue
                data.rows.append(row)
    return data


def _best_cells_per_interpreter(
    data: ReportData, top_n: int = 3
) -> dict[str, list[tuple[tuple[str, str], int, int, float]]]:
    """Rank (model, variant) cells per interpreter by total score.

    Returns a dict mapping interpreter name to a list of
    ((model, variant), total_score, n_rows, median_latency) tuples.
    """
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"score": 0, "rows": 0, "latencies": []})
    )
    for r in data.rows:
        interp = r.get("interpreter", "?")
        key = (r.get("model", "?"), r.get("variant_id", "?"))
        b = buckets[interp][key]
        b["score"] += int(r.get("score", 0))
        b["rows"] += 1
        b["latencies"].append(float(r.get("duration_ms_generation", 0.0)))

    out: dict[str, list[tuple[tuple[str, str], int, int, float]]] = {}
    for interp, keyed in buckets.items():
        ranked = sorted(
            (
                (k, v["score"], v["rows"],
                 statistics.median(v["latencies"]) if v["latencies"] else 0.0)
                for k, v in keyed.items()
            ),
            key=lambda t: (-t[1], t[3]),
        )
        out[interp] = ranked[:top_n]
    return out


def _failure_modes(data: ReportData, top_n: int = 5) -> list[tuple[str, int]]:
    """Count the most common gate/exec error strings."""
    counter: dict[str, int] = defaultdict(int)
    for r in data.rows:
        for err in r.get("gate_errors", []) or []:
            counter[err] += 1
        exec_err = r.get("exec_error")
        if exec_err:
            counter[exec_err] += 1
    ranked = sorted(counter.items(), key=lambda kv: -kv[1])
    return ranked[:top_n]


def build_report(data: ReportData) -> str:
    """Render the markdown report from consolidated data."""
    lines: list[str] = []
    lines.append("# Prompt Eval Report")
    lines.append("")
    lines.append(f"- Sources: {', '.join(data.sources)}")
    lines.append(f"- Toybox hash: `{data.toybox_hash}`")
    lines.append(f"- Total rows: {len(data.rows)}")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    best = _best_cells_per_interpreter(data, top_n=1)
    lines.append("| Interpreter | Best (model, variant) | Score | Rows | Median gen ms |")
    lines.append("|---|---|---|---|---|")
    for interp in ("python", "ast-select", "pss", "plucker"):
        if interp in best and best[interp]:
            (m, v), score, n, lat = best[interp][0]
            lines.append(f"| {interp} | `{m}` / `{v}` | {score} | {n} | {lat:.0f} |")
        else:
            lines.append(f"| {interp} | — | — | — | — |")
    lines.append("")

    all_best = _best_cells_per_interpreter(data, top_n=3)
    for interp in ("python", "ast-select", "pss", "plucker"):
        lines.append(f"## {interp}")
        lines.append("")
        lines.append("Top 3 cells:")
        lines.append("")
        lines.append("| Rank | Model | Variant | Score | Rows | Median ms |")
        lines.append("|---|---|---|---|---|---|")
        for i, ((m, v), score, n, lat) in enumerate(all_best.get(interp, []), start=1):
            lines.append(f"| {i} | `{m}` | `{v}` | {score} | {n} | {lat:.0f} |")
        lines.append("")

    lines.append("## Top failure modes")
    lines.append("")
    failures = _failure_modes(data)
    if failures:
        for err, n in failures:
            lines.append(f"- `{err}` — {n}×")
    else:
        lines.append("- _no errors recorded_")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = consolidate_jsonls(args.paths)
    md = build_report(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
