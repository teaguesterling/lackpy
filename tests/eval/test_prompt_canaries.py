"""Canary tests for the prompt evaluation harness.

These tests assert that the current best-known (model, variant) per
interpreter still reaches a 2/2 score on a small canary intent set.
They are marked @pytest.mark.slow and require Ollama to be reachable;
they are skipped otherwise so they do not block normal CI.

Update `_CANARIES` after each phase 1b run with the current winners.
The threshold starts loose (>= 50% of canary intents must pass) and
tightens to "all canary intents must pass" once findings are stable.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytest.importorskip("tqdm", reason="tqdm required for eval harness tests")

from scripts.prompt_eval.harness import HarnessConfig, run_harness
from scripts.prompt_eval.intents_ast_select import AST_SELECT_INTENTS
from scripts.prompt_eval.intents_plucker import PLUCKER_INTENTS
from scripts.prompt_eval.intents_pss import PSS_INTENTS
from scripts.prompt_eval.intents_python import PYTHON_INTENTS
from scripts.prompt_eval.query import summarize_jsonl


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"
DEFAULT_HOST = os.environ.get("LACKPY_EVAL_OLLAMA", "http://localhost:11435")


def _ollama_reachable(host: str) -> bool:
    try:
        parsed = urlparse(host)
        with socket.create_connection(
            (parsed.hostname or "localhost", parsed.port or 11434),
            timeout=1.0,
        ):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ollama_reachable(DEFAULT_HOST), reason="Ollama not reachable"),
]


# Populate this dict from phase 1b results. Format:
#   interpreter -> (model, variant_id, [canary_intent_ids])
# Canary intent ids should be a mix of 1 core + 1 stretch + 1 baseline.
_CANARIES: dict[str, tuple[str, str, list[str]]] = {
    # "python": ("qwen2.5-coder:3b", "specialized_fewshot",
    #            ["py.core.01", "py.core.04", "py.stretch.02"]),
    # "ast-select": (...),
    # "pss": (...),
    # "plucker": (...),
}


_ALL = {
    "python": PYTHON_INTENTS,
    "ast-select": AST_SELECT_INTENTS,
    "pss": PSS_INTENTS,
    "plucker": PLUCKER_INTENTS,
}


@pytest.mark.parametrize("interpreter", list(_CANARIES.keys()))
def test_canary(interpreter: str, tmp_path: Path):
    model, variant_id, intent_ids = _CANARIES[interpreter]
    intents = [i for i in _ALL[interpreter] if i.id in intent_ids]
    assert len(intents) == len(intent_ids), f"missing canary intents for {interpreter}"

    output = tmp_path / f"canary-{interpreter}.jsonl"
    cfg = HarnessConfig(
        output_path=output,
        models=[model],
        interpreters=[interpreter],
        variant_ids=[variant_id],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host=DEFAULT_HOST,
        temperature=0.2,
        timeout=60,
    )
    run_harness(cfg)
    summary = summarize_jsonl(output)
    assert summary["total_rows"] == len(intents)
    # Loose threshold for v1: at least half of canaries must score 2
    passes = summary["by_model"][model]["pass_rate_2"]
    assert passes >= 0.5, (
        f"{interpreter} canary regressed: pass-rate={passes:.2%} "
        f"for model={model} variant={variant_id}"
    )
