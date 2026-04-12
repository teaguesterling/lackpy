"""Failure forensics and analysis for prompt eval JSONL data.

Reads a phase JSONL, classifies every non-passing cell into a failure
taxonomy, audits assertions for false negatives, builds a model × intent
heatmap, and outputs actionable recommendations.

Usage:
    python -m scripts.prompt_eval.analyze results/.../phase1a-qualifier.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Failure categories ─────────────────────────────────────────────────

CATEGORIES = {
    "path_prefix": "Program uses 'toybox/' prefix on relative paths",
    "key_hallucination": "Program accesses wrong dict keys (e.g. 'path' instead of 'file')",
    "implement_not_orchestrate": "Program defines functions/classes instead of calling tools",
    "jupyter_confusion": "Program outputs bare tokens (ipynb, py, sql) from Jupyter framing",
    "syntax_artifact": "Program contains non-Python syntax (-> annotations, prose)",
    "stdlib_usage": "Program uses open()/import instead of kit tools",
    "exec_path_error": "Execution failed with file-not-found or path error",
    "exec_key_error": "Execution failed with KeyError on dict access",
    "exec_other_error": "Execution failed with another runtime error",
    "assertion_mismatch": "Execution succeeded but assertion rejected the output",
    "empty_generation": "Model returned empty output",
    "gate_other": "Gate rejected for another reason",
}


def _classify(row: dict) -> str:
    """Assign a single failure category to a non-passing cell."""
    score = row.get("score", 0)
    if score == 2:
        return "pass"

    raw = row.get("raw_generation", "")
    sanitized = row.get("sanitized_program", "")
    gate_passed = row.get("gate_passed", False)
    gate_errors = row.get("gate_errors", []) or []
    exec_error = row.get("exec_error") or ""
    executed = row.get("executed", False)

    if not raw and not sanitized:
        return "empty_generation"

    gate_err_text = " ".join(gate_errors).lower()

    if not gate_passed:
        if "functiondef" in gate_err_text or "classdef" in gate_err_text:
            return "implement_not_orchestrate"
        if "import" in gate_err_text:
            return "implement_not_orchestrate"
        if "forbidden name" in gate_err_text and "open" in gate_err_text:
            return "stdlib_usage"
        if "parse error" in gate_err_text or "invalid syntax" in gate_err_text:
            if "->" in sanitized:
                return "syntax_artifact"
            return "gate_other"
        return "gate_other"

    # Gate passed — check execution
    stripped = sanitized.strip()
    if stripped in ("ipynb", "py", "sql", "python", "jupyter"):
        return "jupyter_confusion"

    if "toybox/" in sanitized or "toybox\\" in sanitized:
        if "no such file" in exec_error.lower() or "errno 2" in exec_error.lower():
            return "path_prefix"

    if exec_error:
        if "no such file" in exec_error.lower() or "errno 2" in exec_error.lower():
            return "exec_path_error"
        if "escapes" in exec_error.lower() and "base_dir" in exec_error.lower():
            return "path_prefix"
        if "keyerror" in exec_error.lower() or ("'" in exec_error and exec_error.count("'") == 2):
            # Check for key hallucination patterns
            for bad_key in ("path", "filename", "name", "body"):
                if f"'{bad_key}'" in exec_error:
                    return "key_hallucination"
            return "exec_key_error"
        if "not defined" in exec_error.lower():
            if any(tok in stripped for tok in ("ipynb", "py ", "sql")):
                return "jupyter_confusion"
            return "exec_other_error"
        return "exec_other_error"

    # Gate passed, execution succeeded, but assertion failed
    if executed and not row.get("assertion_passed", False):
        return "assertion_mismatch"

    return "gate_other"


# ── Analysis report ────────────────────────────────────────────────────


@dataclass
class AnalysisReport:
    """Complete analysis of a JSONL run."""

    path: str
    total_rows: int = 0
    total_pass: int = 0
    total_fail: int = 0

    failure_taxonomy: dict[str, list[dict]] = field(default_factory=dict)
    assertion_audit: list[dict] = field(default_factory=list)
    model_heatmap: dict[str, dict[str, int]] = field(default_factory=dict)
    model_summary: list[dict] = field(default_factory=list)
    prompt_fixes: list[dict] = field(default_factory=list)
    cohort_recommendation: list[str] = field(default_factory=list)


def analyze_run(path: Path) -> AnalysisReport:
    """Analyze a JSONL file and produce a structured report."""
    path = Path(path)
    report = AnalysisReport(path=str(path))

    rows: list[dict] = []
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
                continue
            rows.append(row)

    report.total_rows = len(rows)
    report.total_pass = sum(1 for r in rows if r.get("score") == 2)
    report.total_fail = report.total_rows - report.total_pass

    # ── Failure taxonomy ───────────────────────────────────────────
    taxonomy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cat = _classify(row)
        if cat == "pass":
            continue
        example = {
            "model": row.get("model"),
            "intent_id": row.get("intent_id"),
            "score": row.get("score"),
            "sanitized_program": (row.get("sanitized_program") or "")[:120],
            "exec_error": (row.get("exec_error") or "")[:100],
            "gate_errors": (row.get("gate_errors") or [])[:2],
            "category": cat,
        }
        taxonomy[cat].append(example)
    report.failure_taxonomy = dict(taxonomy)

    # ── Assertion audit (score 1 with successful execution) ────────
    for row in rows:
        if (row.get("score") == 1
                and row.get("executed")
                and row.get("gate_passed")
                and not row.get("exec_error")):
            report.assertion_audit.append({
                "model": row.get("model"),
                "intent_id": row.get("intent_id"),
                "exec_output_repr": (row.get("exec_output_repr") or "")[:200],
                "sanitized_program": (row.get("sanitized_program") or "")[:120],
            })

    # ── Model × intent heatmap ─────────────────────────────────────
    heatmap: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        model = row.get("model", "?")
        intent = row.get("intent_id", "?")
        heatmap[model][intent] = row.get("score", 0)
    report.model_heatmap = dict(heatmap)

    # ── Model summary ──────────────────────────────────────────────
    model_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "pass2": 0, "pass1": 0, "fail0": 0,
                 "categories": Counter(), "latencies": []}
    )
    for row in rows:
        model = row.get("model", "?")
        s = model_stats[model]
        score = row.get("score", 0)
        s["total"] += 1
        if score == 2:
            s["pass2"] += 1
        elif score == 1:
            s["pass1"] += 1
        else:
            s["fail0"] += 1
        cat = _classify(row)
        if cat != "pass":
            s["categories"][cat] += 1
        s["latencies"].append(row.get("duration_ms_generation", 0))

    import statistics
    summaries = []
    for model, s in sorted(model_stats.items(),
                           key=lambda kv: -kv[1]["pass2"]):
        top_failures = s["categories"].most_common(3)
        summaries.append({
            "model": model,
            "total": s["total"],
            "pass2": s["pass2"],
            "pass1": s["pass1"],
            "fail0": s["fail0"],
            "pass2_rate": s["pass2"] / s["total"] if s["total"] else 0,
            "median_ms": statistics.median(s["latencies"]) if s["latencies"] else 0,
            "top_failures": top_failures,
        })
    report.model_summary = summaries

    # ── Prompt fix recommendations ─────────────────────────────────
    cat_counts = Counter()
    for cat, examples in taxonomy.items():
        cat_counts[cat] = len(examples)

    fixes = []
    if cat_counts.get("path_prefix", 0) > 0:
        fixes.append({
            "category": "path_prefix",
            "count": cat_counts["path_prefix"],
            "fix": "Reinforce in prompt: 'All paths are relative. Use app.py not toybox/app.py'",
            "where": "Already in namespace_desc; may need stronger emphasis in specialized variant",
        })
    if cat_counts.get("key_hallucination", 0) > 0:
        fixes.append({
            "category": "key_hallucination",
            "count": cat_counts["key_hallucination"],
            "fix": "Document return schema explicitly: 'find_def returns dicts with keys: file, line, text'",
            "where": "namespace_desc in eval_kit.py (already fixed) + specialized prompt examples",
        })
    if cat_counts.get("implement_not_orchestrate", 0) > 0:
        fixes.append({
            "category": "implement_not_orchestrate",
            "count": cat_counts["implement_not_orchestrate"],
            "fix": "Stronger framing: 'CALL the tools, do NOT define functions'",
            "where": "specialized variant targets this directly",
        })
    if cat_counts.get("jupyter_confusion", 0) > 0:
        fixes.append({
            "category": "jupyter_confusion",
            "count": cat_counts["jupyter_confusion"],
            "fix": "Drop Jupyter-cell framing for non-code-completion models",
            "where": "specialized variant uses non-Jupyter framing",
        })
    if cat_counts.get("assertion_mismatch", 0) > 0:
        fixes.append({
            "category": "assertion_mismatch",
            "count": cat_counts["assertion_mismatch"],
            "fix": "Review assertion_audit entries — some may be correct outputs rejected by tight assertions",
            "where": "intents_*.py exec_assertion callables",
        })
    report.prompt_fixes = fixes

    # ── Cohort recommendation ──────────────────────────────────────
    viable = [
        s for s in summaries
        if s["pass2"] > 0 or s["pass2_rate"] >= 0.0
    ]
    # Sort by pass2 desc, then median_ms asc
    viable.sort(key=lambda s: (-s["pass2"], s["median_ms"]))
    report.cohort_recommendation = [s["model"] for s in viable[:8]]

    return report


# ── Display ────────────────────────────────────────────────────────────


def print_analysis(report: AnalysisReport) -> None:
    """Print a human-readable analysis report."""
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {report.path}")
    print(f"{'=' * 70}")
    print(f"Total: {report.total_rows}  Pass(2): {report.total_pass}  Fail: {report.total_fail}")
    print(f"Pass rate: {report.total_pass / report.total_rows * 100:.1f}%")

    print(f"\n── Failure Taxonomy ──")
    for cat in sorted(report.failure_taxonomy.keys(),
                      key=lambda c: -len(report.failure_taxonomy[c])):
        examples = report.failure_taxonomy[cat]
        desc = CATEGORIES.get(cat, cat)
        print(f"\n  {cat} ({len(examples)}×) — {desc}")
        for ex in examples[:3]:
            prog = ex["sanitized_program"].replace("\n", " | ")[:80]
            err = ex.get("exec_error", "") or (ex.get("gate_errors", [""])[0] if ex.get("gate_errors") else "")
            print(f"    {ex['model']:<25} {ex['intent_id']:<14} s={ex['score']} | {prog}")
            if err:
                print(f"      err: {err[:70]}")

    if report.assertion_audit:
        print(f"\n── Assertion Audit ({len(report.assertion_audit)} cells executed OK but assertion failed) ──")
        for entry in report.assertion_audit[:10]:
            prog = entry["sanitized_program"].replace("\n", " | ")[:80]
            output = entry["exec_output_repr"][:100]
            print(f"  {entry['model']:<25} {entry['intent_id']:<14}")
            print(f"    prog: {prog}")
            print(f"    output: {output}")

    print(f"\n── Model Summary ──")
    print(f"  {'Model':<30} {'Pass2':>5} {'Pass1':>5} {'Fail0':>5} {'Rate':>6} {'Med ms':>7}  Top failures")
    for s in report.model_summary:
        top_f = ", ".join(f"{c}({n})" for c, n in s["top_failures"][:2])
        print(f"  {s['model']:<30} {s['pass2']:>5} {s['pass1']:>5} {s['fail0']:>5} "
              f"{s['pass2_rate']:>5.0%} {s['median_ms']:>7.0f}  {top_f}")

    if report.prompt_fixes:
        print(f"\n── Prompt Fix Recommendations ──")
        for fix in report.prompt_fixes:
            print(f"  [{fix['count']}×] {fix['category']}: {fix['fix']}")
            print(f"        where: {fix['where']}")

    print(f"\n── Cohort Recommendation (top 8) ──")
    for i, m in enumerate(report.cohort_recommendation, 1):
        print(f"  {i}. {m}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt eval JSONL results")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    report = analyze_run(args.path)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print_analysis(report)


if __name__ == "__main__":
    main()
