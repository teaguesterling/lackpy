#!/usr/bin/env python3
"""Standalone harness for testing the literate interpreter with Ollama models.

Usage:
    python scripts/literate_agent.py "Analyze the files in src/"
    python scripts/literate_agent.py --model qwen3:8b "List Python files"
    python scripts/literate_agent.py --persona analyst "Find all TODOs"
    python scripts/literate_agent.py --base-dir /some/path "Read README.md"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.prompts import DEFAULT_PERSONA, PERSONAS, compose


DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"


async def _progress_indicator(label: str) -> None:
    """Print dots to stderr while waiting for a response."""
    print(f"  {label} ", end="", file=sys.stderr, flush=True)
    try:
        while True:
            await asyncio.sleep(3)
            print(".", end="", file=sys.stderr, flush=True)
    except asyncio.CancelledError:
        print(file=sys.stderr)


async def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = "",
    num_predict: int = 8192,
    verbose: bool = False,
) -> str:
    """Call Ollama and return the response text."""
    indicator = asyncio.create_task(_progress_indicator("Generating")) if verbose else None
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": num_predict,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]
    finally:
        if indicator:
            indicator.cancel()
            try:
                await indicator
            except asyncio.CancelledError:
                pass
            elapsed = time.perf_counter() - start
            print(f"  ({elapsed:.1f}s)", file=sys.stderr)


async def run_literate_agent(
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    base_dir: str | None = None,
    max_iterations: int = 3,
    verbose: bool = False,
    persona: str = DEFAULT_PERSONA,
    num_predict: int = 8192,
) -> str:
    """Run the literate agent loop.

    1. Send prompt to model, get literate document back
    2. Execute the document
    3. If @continue was hit, feed results back to model
    4. Repeat until no more @continue or max iterations
    """
    interpreter = LiterateInterpreter()
    system_prompt = compose(persona, interpreter)
    work_dir = Path(base_dir) if base_dir else Path.cwd()
    context = ExecutionContext(base_dir=work_dir)

    if verbose:
        print(f"Persona: {persona}", file=sys.stderr)
        print(f"System prompt: {len(system_prompt)} chars", file=sys.stderr)

    full_prompt = (
        f"Task: {user_prompt}\n\n"
        f"Working directory: {work_dir}\n"
        f"Write your response as a literate document."
    )

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Iteration {iteration + 1}/{max_iterations}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

        response = await call_ollama(full_prompt, model=model, system=system_prompt, num_predict=num_predict, verbose=verbose)

        if verbose:
            print(f"\n--- Model Response ---", file=sys.stderr)
            print(response, file=sys.stderr)
            print(f"--- End Response ---\n", file=sys.stderr)

        result = await interpreter.execute(response, context)

        if verbose:
            print(f"\n--- Execution Result ---", file=sys.stderr)
            print(f"Success: {result.success}", file=sys.stderr)
            if result.error:
                print(f"Error: {result.error}", file=sys.stderr)
            print(f"--- End Result ---\n", file=sys.stderr)

        if not result.success:
            return f"[Execution failed: {result.error}]\n\nRaw response:\n{response}"

        if not result.metadata.get("continue_requested"):
            return result.output

        variables = result.metadata.get("variables", {})
        var_summary = "\n".join(f"  {k} = {repr(v)[:200]}" for k, v in variables.items())

        full_prompt = (
            f"Previous execution output:\n{result.output}\n\n"
            f"Variables available:\n{var_summary}\n\n"
            f"Continue writing the document from where @continue left off."
        )

    return result.output


def main():
    parser = argparse.ArgumentParser(description="Literate agent harness")
    parser.add_argument("prompt", help="User prompt for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-dir", default=None, help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max gather/continue iterations")
    parser.add_argument(
        "--persona", default=DEFAULT_PERSONA,
        choices=sorted(PERSONAS),
        help=f"System prompt persona (default: {DEFAULT_PERSONA})",
    )
    parser.add_argument("--num-predict", type=int, default=8192, help="Max tokens for model response (default: 8192)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show model responses on stderr")
    args = parser.parse_args()

    output = asyncio.run(run_literate_agent(
        args.prompt,
        model=args.model,
        base_dir=args.base_dir,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        persona=args.persona,
        num_predict=args.num_predict,
    ))
    print(output)


if __name__ == "__main__":
    main()
