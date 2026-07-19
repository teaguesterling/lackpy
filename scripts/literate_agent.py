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
import json
import sys
from pathlib import Path

import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import (
    CONTINUE_MARKER,
    LiterateInterpreter,
    LiterateSession,
    StopScanner,
)
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
    stop: list[str] | None = None,
) -> str:
    """Call Ollama and return the response text.

    ``stop``: optional stop sequences, scanned CLIENT-SIDE over a streaming
    response via :class:`StopScanner` (NOT the API-native ``options.stop`` --
    native stop strips the matched marker and is indistinguishable from a
    natural end of turn, and it fires inside <think> blocks; see StopScanner).
    On a hit the HTTP stream is closed, which makes Ollama cancel generation;
    the returned text ends with (and includes) the matched marker so the
    session's textual fallback owns the pause semantics.
    """
    indicator = asyncio.create_task(_progress_indicator("Generating")) if verbose else None
    start = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": bool(stop),
        "options": {
            "temperature": 0.7,
            "num_predict": num_predict,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            if not stop:
                response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
                response.raise_for_status()
                return response.json()["response"]

            scanner = StopScanner(stop)
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if scanner.feed(data.get("response", "")):
                        break  # closing the stream cancels generation
                    if data.get("done"):
                        break
            return scanner.text
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
    # The fold. One persistent kernel is threaded across rounds, so functions and
    # objects defined in one round are live in the next; step() strips <think>,
    # gates + journals the doc, and returns Either. This thin client just does the
    # model-call loop and feeds the next round.
    # max_rounds surfaces THIS loop's real budget (max_iterations) to the
    # writer via the L4 per-splice manifest; the loop below still enforces it.
    session = LiterateSession(
        context, interpreter=interpreter, max_rounds=max_iterations
    )

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

        # Stop generation at the pause marker (client-side scan; the marker is
        # kept and the session's textual fallback handles the pause + discard).
        response = await call_ollama(
            full_prompt, model=model, system=system_prompt,
            num_predict=num_predict, verbose=verbose,
            stop=[CONTINUE_MARKER],
        )

        if verbose:
            print("\n--- Model Response ---", file=sys.stderr)
            print(response, file=sys.stderr)
            print("--- End Response ---\n", file=sys.stderr)

        result = await session.step(response)

        if verbose:
            print("\n--- Step Result ---", file=sys.stderr)
            print(f"ok: {result.ok}", file=sys.stderr)
            if not result.ok:
                print(f"Errors: {result.errors}", file=sys.stderr)
            print("--- End Result ---\n", file=sys.stderr)

        if not result.ok:
            # Left: hand the errors + the raw un-interpreted document back so the
            # model corrects its own work. State is NOT advanced; retry this round.
            errors = "\n".join(result.errors)
            full_prompt = (
                f"Your previous document failed:\n{errors}\n\n"
                f"Here is what you wrote:\n{result.raw}\n\n"
                f"Fix the problem and resubmit the FULL corrected document."
            )
            continue

        if not result.continue_requested:
            return session.rendered

        # Right + @continue: feed back the CANONICAL source-preserving document
        # (session.rendered) — round-trippable, kernel notes inert — NOT the
        # flat stdout (result.clean_doc), which re-prints/stacks kernel strings
        # when fed forward (the exp1 poisoning L2 closed). Live values reach the
        # model through session.scope (real objects already in the kernel), not
        # by interpolating them into the fed-back document.
        scope_summary = "\n".join(f"  {k} = {v}" for k, v in session.scope.items())
        full_prompt = (
            f"Document so far:\n{session.rendered}\n\n"
            f"Live variables (already in scope — do not redefine):\n{scope_summary}\n\n"
            f"Continue writing the document from where @continue left off."
        )

    return session.rendered


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
