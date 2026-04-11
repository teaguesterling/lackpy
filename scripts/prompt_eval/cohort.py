"""Model cohort lists and selection helpers for phase-to-phase handoff."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Phase 1a cohort: all useful Ollama models on localhost:11435.
# Sorted by size (ascending) so we go small-to-large and can bail
# early if memory pressure hits. Vision models are excluded.
PHASE1A_MODELS: list[str] = [
    # Tiny (<1GB)
    "qwen2.5-coder:0.5b",
    "qwen3:0.6b",
    # Small (1-2GB)
    "llama3.2:1b",
    "qwen2.5-coder:1.5b",
    "qwen2.5:1.5b",
    "codegemma:2b",
    "granite3.1-dense:2b",
    "granite3.3:2b",
    "smollm2:latest",
    # Medium (2-3GB)
    "llama3.2:latest",
    "phi4-mini:latest",
    "qwen2.5-coder:3b",
    "qwen2.5:3b",
    "granite-code:3b",
    "phi3:latest",
    # Large (5-8GB)
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "qwen2:latest",
    # Very large (9GB)
    "gemma:latest",
]


def pick_phase1b_cohort(
    phase1a_jsonl: Path,
    top_n: int = 6,
    gate_floor: float = 0.5,
) -> list[str]:
    """Select the top-N models from a phase 1a qualifier JSONL.

    Rules:
      - Drop models whose gate-pass rate is below `gate_floor`.
      - Rank survivors by total score (descending).
      - Ties broken by median generation latency (ascending).

    Returns:
        A list of model ids ordered best-first.
    """
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total_score": 0,
        "total_rows": 0,
        "gate_passes": 0,
        "latencies": [],
    })
    with phase1a_jsonl.open() as f:
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
            model = row.get("model")
            if not model:
                continue
            b = by_model[model]
            b["total_rows"] += 1
            b["total_score"] += int(row.get("score", 0))
            if row.get("gate_passed"):
                b["gate_passes"] += 1
            b["latencies"].append(float(row.get("duration_ms_generation", 0.0)))

    survivors: list[tuple[str, int, float]] = []
    for model, b in by_model.items():
        if b["total_rows"] == 0:
            continue
        gate_rate = b["gate_passes"] / b["total_rows"]
        if gate_rate < gate_floor:
            continue
        median_latency = statistics.median(b["latencies"]) if b["latencies"] else 0.0
        survivors.append((model, b["total_score"], median_latency))

    survivors.sort(key=lambda t: (-t[1], t[2]))
    return [m for m, _s, _l in survivors[:top_n]]
