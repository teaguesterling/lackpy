"""Configuration recommender based on accumulated trial and phase data.

Queries the trial log (results/trials.jsonl) and any phase sweep JSONLs
to recommend the best (model, variant, interpreter) for a given task.

Usage:
    python -m scripts.prompt_eval.advisor "find all functions that call execute_sql"
    python -m scripts.prompt_eval.advisor "show the User class" --interpreter ast-select
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Recommendation:
    """A recommended configuration for a task."""

    model: str
    variant: str
    interpreter: str
    confidence: float
    based_on: int
    supporting_evidence: list[dict] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from a task description."""
    text = text.lower()
    tokens = set(re.findall(r"[a-z_][a-z0-9_]*", text))
    # Remove very common words
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "to", "for",
                 "is", "it", "as", "on", "at", "by", "from", "with",
                 "that", "this", "be", "are", "was", "were", "all",
                 "every", "each", "return", "them", "their", "its"}
    return tokens - stopwords


def _overlap_score(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard-like overlap between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _load_trials(
    trial_log: Path,
    phase_jsonls: list[Path] | None = None,
) -> list[dict]:
    """Load all trial records from the log + any phase JSONL files."""
    records: list[dict] = []

    if trial_log.exists():
        with trial_log.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "_meta" not in row and "task" in row:
                        records.append(row)
                except json.JSONDecodeError:
                    continue

    for p in (phase_jsonls or []):
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "_meta" in row:
                        continue
                    # Phase JSONL rows use intent_text instead of task
                    if "intent_text" in row:
                        row["task"] = row["intent_text"]
                    if "task" in row:
                        records.append(row)
                except json.JSONDecodeError:
                    continue

    return records


def recommend(
    task_description: str,
    interpreter: str | None = None,
    trial_log: Path = Path("results/trials.jsonl"),
    phase_jsonls: list[Path] | None = None,
    top_n: int = 3,
    min_overlap: float = 0.15,
) -> Recommendation | None:
    """Recommend the best (model, variant, interpreter) for a task.

    Searches accumulated trial/phase data for tasks with keyword overlap,
    then ranks configurations by win rate on those similar tasks.

    Args:
        task_description: Natural language description of the task.
        interpreter: If set, only consider trials with this interpreter.
        trial_log: Path to the persistent trial log.
        phase_jsonls: Additional phase JSONL files to mine.
        top_n: Number of alternative configs to include.
        min_overlap: Minimum token overlap to consider a trial "similar".

    Returns:
        A Recommendation, or None if no relevant data exists.
    """
    if phase_jsonls is None:
        results_dir = Path("results/prompt-eval-2026-04-11")
        phase_jsonls = sorted(results_dir.glob("*.jsonl")) if results_dir.exists() else []

    records = _load_trials(trial_log, phase_jsonls)
    if not records:
        return None

    query_tokens = _tokenize(task_description)
    if not query_tokens:
        return None

    # Find similar trials
    similar: list[tuple[float, dict]] = []
    for rec in records:
        task_text = rec.get("task", "")
        if not task_text:
            continue
        if interpreter and rec.get("interpreter") != interpreter:
            continue
        rec_tokens = _tokenize(task_text)
        overlap = _overlap_score(query_tokens, rec_tokens)
        if overlap >= min_overlap:
            similar.append((overlap, rec))

    if not similar:
        return None

    # Rank configs by weighted win rate
    config_stats: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"weighted_score": 0.0, "total_weight": 0.0, "count": 0}
    )
    for overlap, rec in similar:
        key = (
            rec.get("model", "?"),
            rec.get("variant", rec.get("variant_id", "?")),
            rec.get("interpreter", "?"),
        )
        score = rec.get("score", 0)
        stats = config_stats[key]
        stats["weighted_score"] += score * overlap
        stats["total_weight"] += overlap * 2  # max score is 2
        stats["count"] += 1

    ranked = sorted(
        config_stats.items(),
        key=lambda kv: -(kv[1]["weighted_score"] / kv[1]["total_weight"]
                         if kv[1]["total_weight"] > 0 else 0),
    )

    if not ranked:
        return None

    best_key, best_stats = ranked[0]
    confidence = (best_stats["weighted_score"] / best_stats["total_weight"]
                  if best_stats["total_weight"] > 0 else 0)

    # Gather supporting evidence
    evidence = []
    for overlap, rec in sorted(similar, key=lambda t: -t[0])[:5]:
        if (rec.get("model"), rec.get("variant", rec.get("variant_id")),
                rec.get("interpreter")) == best_key:
            evidence.append({
                "task": rec.get("task", "")[:80],
                "score": rec.get("score", 0),
                "overlap": round(overlap, 2),
            })

    # Alternatives
    alternatives = []
    for key, stats in ranked[1:top_n]:
        conf = (stats["weighted_score"] / stats["total_weight"]
                if stats["total_weight"] > 0 else 0)
        alternatives.append({
            "model": key[0],
            "variant": key[1],
            "interpreter": key[2],
            "confidence": round(conf, 2),
            "based_on": stats["count"],
        })

    return Recommendation(
        model=best_key[0],
        variant=best_key[1],
        interpreter=best_key[2],
        confidence=round(confidence, 2),
        based_on=best_stats["count"],
        supporting_evidence=evidence,
        alternatives=alternatives,
    )


def print_recommendation(rec: Recommendation | None) -> None:
    """Print a formatted recommendation."""
    if rec is None:
        print("No recommendation available — not enough historical data.")
        return

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"  Model:       {rec.model}")
    print(f"  Variant:     {rec.variant}")
    print(f"  Interpreter: {rec.interpreter}")
    print(f"  Confidence:  {rec.confidence:.0%}")
    print(f"  Based on:    {rec.based_on} similar trial(s)")

    if rec.supporting_evidence:
        print(f"\n  Supporting evidence:")
        for ev in rec.supporting_evidence[:3]:
            print(f"    score={ev['score']} overlap={ev['overlap']} | {ev['task']}")

    if rec.alternatives:
        print(f"\n  Alternatives:")
        for alt in rec.alternatives:
            print(f"    {alt['model']} / {alt['variant']} / {alt['interpreter']} "
                  f"— {alt['confidence']:.0%} ({alt['based_on']} trials)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a model/prompt config for a task")
    parser.add_argument("task", help="Natural language task description")
    parser.add_argument("--interpreter", default=None,
                        choices=["python", "ast-select", "pss", "plucker"])
    parser.add_argument("--trial-log", type=Path, default=Path("results/trials.jsonl"))
    args = parser.parse_args()

    rec = recommend(
        task_description=args.task,
        interpreter=args.interpreter,
        trial_log=args.trial_log,
    )
    print_recommendation(rec)


if __name__ == "__main__":
    main()
