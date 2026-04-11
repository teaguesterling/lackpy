"""Live JSONL summary for the prompt eval harness.

Usage:
    python -m scripts.prompt_eval.query <path-to-jsonl>

Can be run while the harness is still writing. Reads the file once
and prints per-dimension aggregates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _fresh_group() -> dict[str, Any]:
    return {"rows": 0, "sum_score": 0, "pass_rate_2": 0.0, "latencies": []}


def summarize_jsonl(path: Path) -> dict[str, Any]:
    """Aggregate a JSONL file into by-model, by-variant, by-interpreter dicts.

    Missing files, `_meta`-only files, and malformed rows are all
    handled gracefully: the summary stays consistent and non-empty
    even in degenerate cases.
    """
    summary: dict[str, Any] = {
        "path": str(path),
        "total_rows": 0,
        "meta": None,
        "by_model": {},
        "by_variant": {},
        "by_interpreter": {},
        "global_median_ms": 0.0,
    }
    if not path.exists():
        return summary
    all_latencies: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                summary["meta"] = row["_meta"]
                continue
            summary["total_rows"] += 1
            score = int(row.get("score", 0))
            latency = float(row.get("duration_ms_generation", 0.0))
            all_latencies.append(latency)
            for bucket, key in (
                ("by_model", row.get("model", "?")),
                ("by_variant", row.get("variant_id", "?")),
                ("by_interpreter", row.get("interpreter", "?")),
            ):
                g = summary[bucket].setdefault(key, _fresh_group())
                g["rows"] += 1
                g["sum_score"] += score
                g["latencies"].append(latency)
                if score == 2:
                    g["pass_rate_2"] += 1
    for bucket in ("by_model", "by_variant", "by_interpreter"):
        for _key, g in summary[bucket].items():
            if g["rows"]:
                g["pass_rate_2"] = g["pass_rate_2"] / g["rows"]
                g["median_latency_ms"] = statistics.median(g["latencies"])
            else:
                g["median_latency_ms"] = 0.0
            # Drop the raw latencies list from the output
            del g["latencies"]
    if all_latencies:
        summary["global_median_ms"] = statistics.median(all_latencies)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n== {summary['path']} ==")
    print(
        f"total rows: {summary['total_rows']}    "
        f"global median latency: {summary['global_median_ms']:.0f}ms"
    )
    if summary["meta"]:
        meta = summary["meta"]
        created = meta.get("created_utc", "?")
        h = (meta.get("toybox_hash") or "?")[:16]
        print(f"toybox hash: {h}…  created: {created}")
    for bucket_name in ("by_model", "by_interpreter", "by_variant"):
        print(f"\n[{bucket_name}]")
        items = sorted(
            summary[bucket_name].items(),
            key=lambda kv: -kv[1]["sum_score"],
        )
        for key, g in items:
            print(
                f"  {key:<40} rows={g['rows']:<4} "
                f"score_sum={g['sum_score']:<4} "
                f"pass2={g['pass_rate_2'] * 100:5.1f}%  "
                f"median={g['median_latency_ms']:.0f}ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = summarize_jsonl(args.path)
    print_summary(summary)


if __name__ == "__main__":
    main()
