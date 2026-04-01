#!/usr/bin/env python3
"""Compare models on pluckit chain generation.

Usage:
    python scripts/pluckit-model-compare.py [--host http://localhost:11435] [--models qwen2.5-coder:1.5b,phi4-mini]
"""

import argparse
import json
import time



import ollama

SYSTEM = """You write a single jQuery-style chain for querying and mutating source code ASTs.

Examples:
  select('.fn#validate_token').callers()
  select('.fn:exported').addParam('timeout: int = 30').test().save('feat: add timeout')
  source('src/**/*.py').find('.fn:has(.ret > .none)').count()
  select('.fn#handle_request').call_chain()
  select('.fn').filter(fn => fn.complexity() > 10).filter(fn => fn.coverage() < 0.5)

API (chainable):
  select(css) / source(glob).find(css) - entry points. CSS: .fn .cls .call #name :has() :not() :exported [name^="test_"]
  .find(css) .filter(fn => cond) .callers() .callees() .similar(n) .reachable(n) .call_chain()
  .refs() .defs() .unused_params() .shadows()
  .addParam(spec) .removeParam(name) .rename(name) .prepend(code) .append(code) .wrap(before,after) .extract(name) .refactor(name)
  .guard(type, strategy) .black() .test(inputs?) .save(msg?) .isolate()
  .text() .count() .names() .complexity() .coverage() .interface()
  .history() .at(ref) .diff(other) .blame() .authors() .filmstrip() .when(css)
  .impact() .compare() .intent(desc)

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

ALL_MODELS = [
    # Small (< 1.5GB)
    "qwen2.5-coder:0.5b",
    "qwen3:0.6b",
    "qwen2.5-coder:1.5b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    # Medium (1.5-3GB)
    "codegemma:2b",
    "smollm2:latest",
    "llama3.2:latest",
    "phi4-mini:latest",
    "qwen2.5-coder:3b",
    "qwen2.5:3b",
    "granite-code:3b",
    # Large (3-8GB)
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
]


def score_output(content: str) -> int:
    has_select = "select(" in content or "source(" in content
    has_chain = "." in content and "(" in content
    is_single_expr = "\n" not in content or content.count("\n") <= 1
    no_def = "function " not in content and "def " not in content and "import " not in content
    return sum([has_select, has_chain, is_single_expr, no_def])


def strip_fences(content: str) -> str:
    for fence in ["```javascript", "```js", "```python", "```py", "```typescript", "```ts", "```"]:
        content = content.replace(fence, "")
    return content.strip()


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


def run_model(client, model: str, intents: list[str], timeout: int = 60) -> list[dict]:
    results = []
    for intent in intents:
        start = time.time()
        try:
            resp = _chat_with_timeout(
                client, model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": intent},
                ],
                options={"temperature": 0.2},
                timeout=timeout,
            )
            elapsed = time.time() - start
            content = strip_fences(resp.message.content.strip())
            gen_tokens = getattr(resp, "eval_count", 0) or 0
            prompt_tokens = getattr(resp, "prompt_eval_count", 0) or 0
            sc = score_output(content)

            print(f"  [{sc}/4] {intent[:50]:50s} | {elapsed:.1f}s {gen_tokens:3d}t | {content[:90]}")
            results.append({
                "intent": intent, "output": content,
                "time": elapsed, "tokens": gen_tokens,
                "prompt_tokens": prompt_tokens, "score": sc,
            })
        except TimeoutError:
            elapsed = time.time() - start
            print(f"  [T/O] {intent[:50]:50s} | {elapsed:.0f}s (timeout)")
            results.append({"intent": intent, "output": "TIMEOUT", "time": elapsed, "tokens": 0, "score": 0})
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [ERR] {intent[:50]:50s} | {e}")
            results.append({"intent": intent, "output": str(e), "time": elapsed, "tokens": 0, "score": 0})
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare models on pluckit chain generation")
    parser.add_argument("--host", default="http://localhost:11435", help="Ollama host")
    parser.add_argument("--models", default=None, help="Comma-separated model list (default: all small models)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per intent in seconds (default: 60)")
    parser.add_argument("--output", default=None, help="Save full results as JSON")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else ALL_MODELS
    client = ollama.Client(host=args.host)

    # Check which models are available
    listed = client.list()
    if hasattr(listed, "models"):
        available = {m.model for m in listed.models}
    else:
        available = {m["name"] for m in listed.get("models", [])}
    models = [m for m in models if m in available]
    skipped = [m for m in (args.models.split(",") if args.models else ALL_MODELS) if m not in available]
    if skipped:
        print(f"Skipping unavailable: {skipped}")

    all_results = {}
    for model in models:
        print(f'\n{"#" * 70}')
        print(f"MODEL: {model}")
        print(f'{"#" * 70}')
        model_results = run_model(client, model, INTENTS, timeout=args.timeout)

        avg_score = sum(r["score"] for r in model_results) / len(model_results)
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        avg_tokens = sum(r["tokens"] for r in model_results) / len(model_results)
        print(f"\n  AVG: score={avg_score:.1f}/4  time={avg_time:.1f}s  tokens={avg_tokens:.0f}")
        all_results[model] = {
            "avg_score": avg_score, "avg_time": avg_time,
            "avg_tokens": avg_tokens, "results": model_results,
        }

    print(f'\n\n{"=" * 70}')
    print("SUMMARY")
    print(f'{"=" * 70}')
    print(f'{"Model":30s} {"Score":>8s} {"Time":>8s} {"Tokens":>8s}')
    print("-" * 60)
    for model in models:
        r = all_results[model]
        print(f'{model:30s} {r["avg_score"]:7.1f}/4 {r["avg_time"]:7.1f}s {r["avg_tokens"]:7.0f}')

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
