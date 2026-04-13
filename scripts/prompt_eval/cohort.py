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
    # Accumulate per-model per-variant stats so we can rank by the
    # model's BEST variant, not aggregate. This prevents models that
    # need examples (coder models) from being penalized when the
    # qualifier also tests baseline.
    by_model_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "score": 0, "rows": 0, "gate_passes": 0, "latencies": [],
        })
    )
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
            variant = row.get("variant_id", "baseline")
            if not model:
                continue
            b = by_model_variant[model][variant]
            b["rows"] += 1
            b["score"] += int(row.get("score", 0))
            if row.get("gate_passed"):
                b["gate_passes"] += 1
            b["latencies"].append(float(row.get("duration_ms_generation", 0.0)))

    survivors: list[tuple[str, int, float]] = []
    for model, variants in by_model_variant.items():
        # Pick the model's best variant by score
        best_score = 0
        best_gate_rate = 0.0
        all_latencies: list[float] = []
        for _variant, b in variants.items():
            if b["rows"] == 0:
                continue
            best_score = max(best_score, b["score"])
            gate_rate = b["gate_passes"] / b["rows"]
            best_gate_rate = max(best_gate_rate, gate_rate)
            all_latencies.extend(b["latencies"])
        if best_gate_rate < gate_floor:
            continue
        median_latency = statistics.median(all_latencies) if all_latencies else 0.0
        survivors.append((model, best_score, median_latency))

    survivors.sort(key=lambda t: (-t[1], t[2]))
    return [m for m, _s, _l in survivors[:top_n]]
