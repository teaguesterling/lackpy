#!/usr/bin/env python3
"""Quartermaster experiment: tool selection → chain assembly.

Three QM modes:
  rules   — keyword matching, zero cost, deterministic (Tier 1)
  model   — LLM selects ops with few-shot examples (Tier 2)
  none    — full API, no QM (baseline)

Usage:
    python scripts/pluckit-quartermaster.py --qm-mode rules --asm-model qwen2.5:1.5b
    python scripts/pluckit-quartermaster.py --qm-mode model --qm-model qwen2.5-coder:0.5b
    python scripts/pluckit-quartermaster.py --qm-mode rules --baseline
"""

import argparse
import json
import re
import time


import ollama


class _ChatResult:
    """Mimics the shape of ollama ChatResponse for streaming results."""
    def __init__(self, content, eval_count, prompt_eval_count):
        self.message = type("Msg", (), {"content": content})()
        self.eval_count = eval_count
        self.prompt_eval_count = prompt_eval_count


def _chat_with_timeout(client, model, messages, options, timeout):
    """Stream a chat call and bail if timeout is exceeded."""
    start = time.time()
    chunks = []
    eval_count = 0
    prompt_eval_count = 0
    for chunk in client.chat(model=model, messages=messages, options=options, stream=True):
        if time.time() - start > timeout:
            raise TimeoutError(f"Exceeded {timeout}s")
        token = chunk.message.content or ""
        chunks.append(token)
        eval_count = getattr(chunk, "eval_count", 0) or eval_count
        prompt_eval_count = getattr(chunk, "prompt_eval_count", 0) or prompt_eval_count
    return _ChatResult("".join(chunks), eval_count, prompt_eval_count)


# =========================================================================
# Rule-based quartermaster (Tier 1: deterministic, zero cost)
# =========================================================================

# Always included — every chain starts with a selection
_BASE_OPS = {"select"}

# Keyword → operations mapping. Each rule adds operations when keywords match.
_QM_RULES: list[tuple[list[str], set[str]]] = [
    # Query patterns
    (["caller", "who calls", "called by"],         {"callers"}),
    (["callee", "calls what", "what does.*call"],   {"callees"}),
    (["similar", "like this", "looks like", "clone"], {"similar"}),
    (["compare", "comparison", "diff between"],     {"compare"}),
    (["call chain", "pipeline", "execution order", "call path"], {"call_chain"}),
    (["reachable", "connected to", "graph"],        {"reachable"}),
    (["impact", "break", "blast radius", "affect"], {"impact"}),
    (["unused param", "unreferenced param"],        {"unused_params"}),
    (["shadow", "shadowing"],                       {"shadows"}),
    (["dead code", "no.*call", "uncalled", "orphan"], {"filter", "callers"}),
    (["complex", "complexity"],                     {"filter", "complexity"}),
    (["coverage", "untested", "uncovered"],         {"filter", "coverage"}),
    (["reference", "who uses"],                     {"references"}),
    (["depend"],                                    {"dependents", "dependencies"}),

    # Scope / reading
    (["scope", "interface", "reads.*writes"],       {"interface"}),
    (["isolat", "runnable", "independently"],       {"isolate"}),
    (["text", "source", "show me", "display"],      {"text"}),
    (["count", "how many"],                         {"count"}),
    (["names", "list.*name"],                       {"names"}),

    # Mutations
    (["add param", "add.*parameter", "add.*argument"], {"addParam"}),
    (["remove param", "remove.*parameter"],         {"removeParam"}),
    (["rename"],                                    {"rename"}),
    (["prepend", "add.*top", "insert.*beginning"],  {"prepend"}),
    (["append", "add.*bottom", "add.*end"],         {"append"}),
    (["wrap", "surround"],                          {"wrap"}),
    (["extract", "pull out", "factor out"],         {"extract"}),
    (["refactor", "common pattern", "generalize"],  {"similar", "refactor"}),
    (["inline", "expand"],                          {"inline"}),
    (["error handl", "guard", "try.*except", "catch"], {"guard"}),
    (["move", "relocate"],                          {"source", "find"}),

    # Formatting / testing / saving
    (["format", "black", "prettier"],               {"black"}),
    (["ruff", "lint", "fix lint"],                  {"ruff_fix"}),
    (["isort", "sort import"],                      {"isort"}),
    (["test", "verify", "check"],                   {"test"}),
    (["save", "commit"],                            {"save"}),
    (["intent", "reason", "why"],                   {"intent"}),

    # History
    (["history", "evolution", "over time"],         {"history"}),
    (["version", "at.*time", "last.*build", "before", "ago"], {"at"}),
    (["diff", "changed", "what changed"],           {"diff", "at"}),
    (["blame", "who wrote", "who changed"],         {"blame"}),
    (["author"],                                    {"authors"}),
    (["filmstrip", "timeline", "snapshots"],        {"filmstrip"}),
    (["when did", "first appear", "started"],       {"when"}),
    (["co.?change", "always change together", "shotgun"], {"history"}),

    # Multi-file
    (["all.*file", "across.*codebase", "everywhere", "src/"], {"source", "find"}),
    (["module", "package"],                         {"source", "find"}),

    # Filtering
    (["return.*none", "returns none"],              {"find"}),
    (["public", "exported", "private"],             {"find"}),
    (["filter", "where", "with", "that have"],      {"filter"}),
]


def rule_based_qm(intent: str) -> set[str]:
    """Select operations by keyword matching. Zero cost, deterministic."""
    ops = set(_BASE_OPS)
    intent_lower = intent.lower()
    for keywords, operations in _QM_RULES:
        for kw in keywords:
            if re.search(kw, intent_lower):
                ops |= operations
                break
    return ops


# =========================================================================
# Model-based quartermaster (Tier 2: LLM with few-shot examples)
# =========================================================================

FULL_API = """select(css) / source(glob).find(css) - entry. CSS: .fn .cls .call #name :has() :not() :exported [name^="test_"]
.find(css) .filter(fn => cond) .callers() .callees() .similar(n) .reachable(n) .call_chain()
.refs() .defs() .unused_params() .shadows()
.addParam(spec) .removeParam(name) .rename(name) .prepend(code) .append(code) .wrap(before,after) .extract(name) .refactor(name)
.guard(type, strategy) .black() .test(inputs?) .save(msg?) .isolate()
.text() .count() .names() .complexity() .coverage() .interface()
.history() .at(ref) .diff(other) .blame() .authors() .filmstrip() .when(css)
.impact() .compare() .intent(desc)"""

QM_SYSTEM = f"""You select the minimal pluckit operations needed for a code task.

API:
{FULL_API}

Examples:
  "find callers of validate_token" → select, callers
  "add timeout parameter to all exported functions, test, commit" → select, find, addParam, test, save
  "find functions that return None" → select, find, count
  "rename process_data to transform_batch" → select, rename
  "add error handling to all database calls" → select, find, guard, save
  "show what would break if I change validate_token" → select, impact
  "find dead code with no callers" → select, filter, callers, count
  "find complex functions with low coverage" → select, filter, complexity, coverage
  "extract common pattern from validate_ functions" → select, similar, refactor
  "show the call chain from handle_request" → select, call_chain

Output ONLY comma-separated operation names. No explanation."""


ASM_SYSTEM_TEMPLATE = """You write a single jQuery-style chain for querying and mutating source code ASTs.

Examples:
  select('.fn#validate_token').callers()
  select('.fn:exported').addParam('timeout: int = 30').test().save('feat: add timeout')
  source('src/**/*.py').find('.fn:has(.ret > .none)').count()
  select('.fn#handle_request').call_chain()
  select('.fn').filter(fn => fn.complexity() > 10).filter(fn => fn.coverage() < 0.5)
  select('.call[name*="query"]').guard('DatabaseError', 'log and reraise').save('fix: add error handling')
  select('.fn#validate_token').impact()
  select('.fn[name^="validate_"]').similar(0.7).refactor('validate_credential')

Available operations (use ONLY these):
{operations}

CSS selectors: .fn .cls .call #name :has() :not() :exported [name^="prefix"] [name*="substr"]

Output ONLY the chain. No markdown, no fences, no explanation."""

INTENTS = [
    "find all public functions",
    "find callers of validate_token",
    "add a timeout parameter to all exported functions, format, test, and commit",
    "find functions that return None",
    "find similar functions to validate_token and compare them",
    "show the call chain starting from handle_request",
    "add error handling to all database query calls",
    "show what would break if I change validate_token",
    "find dead code that no one calls anymore",
    "extract the common pattern from all validate_ functions into validate_credential",
    "find functions with high complexity and low test coverage",
    "rename process_data to transform_batch everywhere",
]

# Operation descriptions for the assembler prompt
OP_DESCRIPTIONS = {
    "select": "select(css) - select AST nodes by CSS selector",
    "source": "source(glob).find(css) - select from specific files",
    "find": ".find(css) - narrow to descendants",
    "filter": ".filter(fn => cond) - filter nodes by condition",
    "callers": ".callers() - functions that call this",
    "callees": ".callees() - functions this calls",
    "similar": ".similar(threshold) - structurally similar nodes",
    "reachable": ".reachable(depth) - call graph traversal",
    "call_chain": ".call_chain() - linear execution order",
    "refs": ".refs() - name references within selection",
    "defs": ".defs() - name definitions within selection",
    "unused_params": ".unused_params() - unreferenced parameters",
    "shadows": ".shadows() - variables shadowing outer scope",
    "addParam": ".addParam(spec) - add parameter to functions",
    "removeParam": ".removeParam(name) - remove parameter",
    "rename": ".rename(newName) - rename with all references",
    "prepend": ".prepend(code) - insert at top of body",
    "append": ".append(code) - insert at bottom of body",
    "wrap": ".wrap(before, after) - wrap in construct",
    "extract": ".extract(name) - extract into new function",
    "refactor": ".refactor(name) - extract common pattern from similar",
    "guard": ".guard(type, strategy) - context-aware error handling",
    "black": ".black() - format with black",
    "ruff_fix": ".ruff_fix() - auto-fix with ruff",
    "isort": ".isort() - sort imports",
    "test": ".test(inputs?) - run in isolation",
    "save": ".save(msg?) - commit via jetsam",
    "isolate": ".isolate() - make block independently runnable",
    "text": ".text() - get source text",
    "count": ".count() - count nodes",
    "names": ".names() - get node names",
    "complexity": ".complexity() - cyclomatic complexity",
    "coverage": ".coverage() - branch coverage",
    "interface": ".interface() - read/write interface from scope",
    "history": ".history() - all versions of selection",
    "at": ".at(ref) - version at point in time",
    "diff": ".diff(other) - structural diff",
    "blame": ".blame() - per-node attribution",
    "authors": ".authors() - who modified this",
    "filmstrip": ".filmstrip() - version snapshots",
    "when": ".when(css) - when did this property appear?",
    "impact": ".impact() - blast radius view",
    "compare": ".compare() - structural comparison",
    "intent": ".intent(desc) - attach metadata",
}


def strip_fences(content: str) -> str:
    for fence in ["```javascript", "```js", "```python", "```py", "```typescript", "```ts", "```"]:
        content = content.replace(fence, "")
    return content.strip()


def score_output(content: str) -> int:
    has_select = "select(" in content or "source(" in content
    has_chain = "." in content and "(" in content
    is_single_expr = "\n" not in content or content.count("\n") <= 1
    no_def = "function " not in content and "def " not in content and "import " not in content
    return sum([has_select, has_chain, is_single_expr, no_def])


def run_model_qm(client, model: str, intent: str, timeout: int = 60) -> tuple[list[str], float, int]:
    """Model-based QM: LLM selects operations with few-shot examples."""
    start = time.time()
    resp = _chat_with_timeout(
        client, model,
        messages=[
            {"role": "system", "content": QM_SYSTEM},
            {"role": "user", "content": intent},
        ],
        options={"temperature": 0.1},
        timeout=timeout,
    )
    elapsed = time.time() - start
    content = strip_fences(resp.message.content.strip())
    tokens = getattr(resp, "eval_count", 0) or 0
    ops = [op.strip().strip(".").strip("()") for op in content.split(",")]
    ops = [op for op in ops if op and op in OP_DESCRIPTIONS]
    return ops, elapsed, tokens


def run_assembler(client, model: str, intent: str, ops: list[str], timeout: int = 60) -> tuple[str, float, int]:
    """Assembler: write chain using only the selected operations."""
    op_lines = "\n".join(f"  {OP_DESCRIPTIONS[op]}" for op in ops if op in OP_DESCRIPTIONS)
    system = ASM_SYSTEM_TEMPLATE.format(operations=op_lines)
    start = time.time()
    resp = _chat_with_timeout(
        client, model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": intent},
        ],
        options={"temperature": 0.2},
        timeout=timeout,
    )
    elapsed = time.time() - start
    content = strip_fences(resp.message.content.strip())
    tokens = getattr(resp, "eval_count", 0) or 0
    return content, elapsed, tokens


def main():
    parser = argparse.ArgumentParser(description="Quartermaster: tool selection → chain assembly")
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--qm-mode", choices=["rules", "model", "none"], default="rules",
                        help="QM strategy: rules (keyword, $0), model (LLM), none (baseline)")
    parser.add_argument("--qm-model", default="qwen2.5-coder:0.5b", help="Model for QM (when --qm-mode=model)")
    parser.add_argument("--asm-model", default="qwen2.5:1.5b", help="Assembler model")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per call in seconds")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--baseline", action="store_true", help="Also run with full API (no QM) for comparison")
    args = parser.parse_args()

    client = ollama.Client(host=args.host)

    print(f"QM mode:    {args.qm_mode}" + (f" ({args.qm_model})" if args.qm_mode == "model" else ""))
    print(f"Assembler:  {args.asm_model}")
    if args.baseline:
        print(f"Baseline:   {args.asm_model} (full API, no QM)")
    print()

    results = []
    for intent in INTENTS:
        print(f'{"=" * 70}')
        print(f"INTENT: {intent}")

        # Stage 1: Select operations
        qm_time = 0
        qm_tokens = 0
        asm_time = 0

        if args.qm_mode == "rules":
            ops = sorted(rule_based_qm(intent))
            print(f"  QM-RULES ({len(ops)} ops): {', '.join(ops)}")
        elif args.qm_mode == "model":
            try:
                ops, qm_time, qm_tokens = run_model_qm(client, args.qm_model, intent, timeout=args.timeout)
                print(f"  QM-MODEL ({qm_time:.1f}s, {qm_tokens}t, {len(ops)} ops): {', '.join(ops)}")
            except TimeoutError:
                print(f"  QM-MODEL TIMEOUT ({args.timeout}s)")
                ops = []
            except Exception as e:
                print(f"  QM-MODEL ERROR: {e}")
                ops = []
        else:  # none
            ops = sorted(OP_DESCRIPTIONS.keys())
            print(f"  QM-NONE (full API, {len(ops)} ops)")

        # Stage 2: Assembler with selected ops
        if ops:
            try:
                chain, asm_time, asm_tokens = run_assembler(client, args.asm_model, intent, ops, timeout=args.timeout)
                sc = score_output(chain)
                total_time = qm_time + asm_time
                print(f"  ASM ({asm_time:.1f}s, {asm_tokens}t) [{sc}/4]: {chain[:100]}")
            except TimeoutError:
                chain = "TIMEOUT"
                sc = 0
                total_time = qm_time + args.timeout
                print(f"  ASM TIMEOUT ({args.timeout}s)")
            except Exception as e:
                chain = f"ERROR: {e}"
                sc = 0
                total_time = qm_time
                print(f"  ASM ERROR: {e}")
        else:
            chain = "NO OPS SELECTED"
            sc = 0
            total_time = 0

        entry = {
            "intent": intent, "qm_mode": args.qm_mode,
            "ops": ops if isinstance(ops, list) else sorted(ops),
            "ops_count": len(ops), "chain": chain,
            "qm_time": qm_time, "asm_time": asm_time,
            "total_time": total_time, "score": sc,
        }

        # Baseline: full API, no QM
        if args.baseline:
            try:
                all_ops = sorted(OP_DESCRIPTIONS.keys())
                bl_chain, bl_time, bl_tokens = run_assembler(client, args.asm_model, intent, all_ops, timeout=args.timeout)
                bl_sc = score_output(bl_chain)
                print(f"  BASE ({bl_time:.1f}s, {bl_tokens}t) [{bl_sc}/4]: {bl_chain[:100]}")
                entry["baseline_chain"] = bl_chain
                entry["baseline_time"] = bl_time
                entry["baseline_score"] = bl_sc
            except TimeoutError:
                print(f"  BASE TIMEOUT ({args.timeout}s)")
            except Exception as e:
                print(f"  BASE ERROR: {e}")

        results.append(entry)
        print()

    # Summary
    qm_results = [r for r in results if r["chain"] not in ("NO OPS SELECTED", "TIMEOUT")]
    if qm_results:
        avg_score = sum(r["score"] for r in qm_results) / len(qm_results)
        avg_time = sum(r["total_time"] for r in qm_results) / len(qm_results)
        avg_ops = sum(r["ops_count"] for r in qm_results) / len(qm_results)
        print(f'{"=" * 70}')
        print(f"SUMMARY: {args.qm_mode} QM → {args.asm_model}")
        print(f"  Score: {avg_score:.1f}/4  Time: {avg_time:.1f}s  Avg ops: {avg_ops:.1f}")

    if args.baseline:
        bl_results = [r for r in results if "baseline_score" in r]
        if bl_results:
            bl_avg = sum(r["baseline_score"] for r in bl_results) / len(bl_results)
            bl_time = sum(r["baseline_time"] for r in bl_results) / len(bl_results)
            print(f"  Baseline: {bl_avg:.1f}/4  Time: {bl_time:.1f}s  Ops: {len(OP_DESCRIPTIONS)}")
            if qm_results:
                delta = avg_score - bl_avg
                print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f} score")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
