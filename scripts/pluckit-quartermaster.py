#!/usr/bin/env python3
"""Quartermaster experiment: one model picks the tools, another writes the chain.

Stage 1 (Quartermaster): Given an intent and the full API, select the minimal
    set of operations needed. Output: comma-separated operation names.

Stage 2 (Assembler): Given the intent and ONLY the selected operations,
    write the chain. Output: a single chain expression.

Usage:
    python scripts/pluckit-quartermaster.py [--host http://localhost:11435]
    python scripts/pluckit-quartermaster.py --qm-model qwen2.5-coder:0.5b --asm-model qwen2.5-coder:1.5b
    python scripts/pluckit-quartermaster.py --asm-model phi4-mini
"""

import argparse
import json
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

FULL_API = """select(css) / source(glob).find(css) - entry. CSS: .fn .cls .call #name :has() :not() :exported [name^="test_"]
.find(css) .filter(fn => cond) .callers() .callees() .similar(n) .reachable(n) .call_chain()
.refs() .defs() .unused_params() .shadows()
.addParam(spec) .removeParam(name) .rename(name) .prepend(code) .append(code) .wrap(before,after) .extract(name) .refactor(name)
.guard(type, strategy) .black() .test(inputs?) .save(msg?) .isolate()
.text() .count() .names() .complexity() .coverage() .interface()
.history() .at(ref) .diff(other) .blame() .authors() .filmstrip() .when(css)
.impact() .compare() .intent(desc)"""

QM_SYSTEM = f"""You are a tool selector. Given a user's intent for code querying/mutation, select the MINIMAL set of pluckit operations needed.

Full API:
{FULL_API}

Output ONLY a comma-separated list of operation names (e.g.: select,callers,filter,count). No explanation."""

ASM_SYSTEM_TEMPLATE = """You write a single jQuery-style chain for querying and mutating source code ASTs.

Examples:
  select('.fn#validate_token').callers()
  select('.fn:exported').addParam('timeout: int = 30').test().save('feat: add timeout')
  source('src/**/*.py').find('.fn:has(.ret > .none)').count()
  select('.fn').filter(fn => fn.complexity() > 10).filter(fn => fn.coverage() < 0.5)

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


def run_quartermaster(client, model: str, intent: str, timeout: int = 60) -> tuple[list[str], float, int]:
    """Stage 1: Select operations for the intent."""
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
    # Parse comma-separated ops
    ops = [op.strip().strip(".").strip("()") for op in content.split(",")]
    ops = [op for op in ops if op and op in OP_DESCRIPTIONS]
    return ops, elapsed, tokens


def run_assembler(client, model: str, intent: str, ops: list[str], timeout: int = 60) -> tuple[str, float, int]:
    """Stage 2: Write chain using only selected operations."""
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
    parser.add_argument("--qm-model", default="qwen2.5-coder:0.5b", help="Quartermaster model (tool selection)")
    parser.add_argument("--asm-model", default="qwen2.5-coder:1.5b", help="Assembler model (chain writing)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per call in seconds (default: 60)")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--baseline", action="store_true", help="Also run assembler without quartermaster (full API)")
    args = parser.parse_args()

    client = ollama.Client(host=args.host)

    print(f"Quartermaster: {args.qm_model}")
    print(f"Assembler:     {args.asm_model}")
    if args.baseline:
        print(f"Baseline:      {args.asm_model} (full API, no quartermaster)")
    print()

    results = []
    for intent in INTENTS:
        print(f'{"=" * 70}')
        print(f"INTENT: {intent}")

        # Stage 1: Quartermaster
        qm_time = 0
        asm_time = 0
        try:
            ops, qm_time, qm_tokens = run_quartermaster(client, args.qm_model, intent, timeout=args.timeout)
            print(f"  QM ({qm_time:.1f}s, {qm_tokens}t): {', '.join(ops)}")
        except TimeoutError:
            print(f"  QM TIMEOUT ({args.timeout}s)")
            ops = []
        except Exception as e:
            print(f"  QM ERROR: {e}")
            ops = []

        # Stage 2: Assembler with selected ops
        if ops:
            try:
                chain, asm_time, asm_tokens = run_assembler(client, args.asm_model, intent, ops, timeout=args.timeout)
                sc = score_output(chain)
                total_time = qm_time + asm_time
                print(f"  ASM ({asm_time:.1f}s, {asm_tokens}t) [{sc}/4]: {chain[:100]}")
                print(f"  TOTAL: {total_time:.1f}s, ops={len(ops)}")
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
            "intent": intent, "ops": ops, "chain": chain,
            "qm_time": qm_time, "asm_time": asm_time,
            "total_time": total_time, "score": sc,
        }

        # Baseline: assembler with full API (no quartermaster)
        if args.baseline:
            try:
                all_ops = list(OP_DESCRIPTIONS.keys())
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
    qm_results = [r for r in results if r["chain"] != "NO OPS SELECTED"]
    if qm_results:
        avg_score = sum(r["score"] for r in qm_results) / len(qm_results)
        avg_time = sum(r["total_time"] for r in qm_results) / len(qm_results)
        avg_ops = sum(len(r["ops"]) for r in qm_results) / len(qm_results)
        print(f'{"=" * 70}')
        print(f"QUARTERMASTER SUMMARY ({args.qm_model} → {args.asm_model})")
        print(f"  Score: {avg_score:.1f}/4  Time: {avg_time:.1f}s  Avg ops selected: {avg_ops:.1f}")

    if args.baseline:
        bl_results = [r for r in results if "baseline_score" in r]
        if bl_results:
            bl_avg_score = sum(r["baseline_score"] for r in bl_results) / len(bl_results)
            bl_avg_time = sum(r["baseline_time"] for r in bl_results) / len(bl_results)
            print(f"  Baseline score: {bl_avg_score:.1f}/4  Time: {bl_avg_time:.1f}s  (full API, no QM)")
            delta = avg_score - bl_avg_score
            print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f} score, "
                  f"{'+' if avg_time - bl_avg_time >= 0 else ''}{avg_time - bl_avg_time:.1f}s time")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
