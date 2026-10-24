"""scripts/literate_agent.py runs each round through the confined runtime.

Literate stepy Task 5 (D10). The model call is a stub: nothing here reaches
Ollama or any endpoint. The policy is loaded with a temporary home and no
managed settings, so no test reads the real ``~/.claude``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lackpy.run.paths import load_policy

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "literate_agent.py"


@pytest.fixture(scope="module")
def agent():
    spec = importlib.util.spec_from_file_location("literate_agent_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def loader(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return lambda ws: load_policy(ws, home=home, managed=None)


def _stub(*docs):
    calls = []
    queue = list(docs)

    async def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return queue.pop(0)

    generate.calls = calls
    return generate


@pytest.mark.asyncio
async def test_a_round_reads_the_workspace_through_path(agent, loader, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "n.txt").write_text("forty-two")
    # The script returns the source-preserving render, so the proof that the
    # cell ran is its effect on the workspace.
    gen = _stub("<compute>\nfrom pathlib import Path\nt = Path('n.txt').read_text()\n"
                "Path('copy.txt').write_text(t)\n</compute>\n")
    await agent.run_literate_agent("copy it", base_dir=str(ws), generate=gen, policy_loader=loader)
    assert (ws / "copy.txt").read_text() == "forty-two"
    assert len(gen.calls) == 1


@pytest.mark.asyncio
async def test_a_refusal_ends_the_document_and_the_loop(agent, loader, tmp_path):
    """Writing .claude/settings.json refuses (gate 2). The loop must not ask
    the model for a fix: the session is closed, every later round refuses."""
    ws = tmp_path / "ws"
    (ws / ".claude").mkdir(parents=True)
    gen = _stub(
        "<compute>\nfrom pathlib import Path\nPath('.claude/settings.json').write_text('{}')\n</compute>\n",
        "should never be asked for",
    )
    out = await agent.run_literate_agent("x", base_dir=str(ws), generate=gen, policy_loader=loader,
                                         max_iterations=3)
    assert "refused" in out
    assert len(gen.calls) == 1
    assert not (ws / ".claude" / "settings.json").exists()


@pytest.mark.asyncio
async def test_the_system_prompt_is_the_stepy_prompt(agent, loader, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    gen = _stub("plain prose\n")
    await agent.run_literate_agent("x", base_dir=str(ws), generate=gen, policy_loader=loader)
    system = gen.calls[0][1]["system"]
    assert "read_file" not in system and "run_command" not in system


def test_the_script_has_no_other_exec_path():
    text = SCRIPT.read_text()
    for word in ("exec(", "eval(", "compile(", "subprocess", "os.system", "LightweightKernel"):
        assert word not in text, word


@pytest.mark.asyncio
async def test_the_cli_shows_the_executed_document(agent, loader, tmp_path):
    """Ruling: a reader sees values, not `{t}` (session.display, not rendered)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "n.txt").write_text("forty-two")
    gen = _stub("<compute hidden>\nfrom pathlib import Path\nt = Path('n.txt').read_text()\n</compute>\n\nGot {t}.\n")
    out = await agent.run_literate_agent("read it", base_dir=str(ws), generate=gen, policy_loader=loader)
    assert "Got forty-two." in out
    assert "{t}" not in out


@pytest.mark.asyncio
async def test_a_corrected_resubmission_runs_in_a_fresh_session(agent, loader, tmp_path):
    """Final review I1, the reviewer's reproduction: round 1 imports, then raises.
    The kept state made the full resubmission fail with "Rebinding of imported
    name" until max_iterations. A full resubmission now starts a new session."""
    ws = tmp_path / "ws"
    ws.mkdir()
    gen = _stub(
        "<compute>\nimport json\nfrom pathlib import Path\nx = 1 // 0\n</compute>\n",
        "<compute>\nimport json\nfrom pathlib import Path\nx = json.dumps([1])\n"
        "Path('out.txt').write_text(x)\n</compute>\n\nGot {x}.\n",
        "never asked",
    )
    out = await agent.run_literate_agent("x", base_dir=str(ws), generate=gen, policy_loader=loader,
                                         max_iterations=3)
    assert len(gen.calls) == 2
    assert "Rebinding" not in gen.calls[1][0]
    assert "ZeroDivisionError" in gen.calls[1][0]
    assert (ws / "out.txt").read_text() == "[1]"
    assert "Got [1]." in out


@pytest.mark.asyncio
async def test_a_resubmission_runs_once_in_a_fresh_session_without_replay(agent, loader, tmp_path):
    """Ruling (after C1): a full-document resubmission runs the resubmitted
    document once, in a fresh session; earlier rounds are not replayed. The
    retry prompt shows the whole document so far, so the model can resubmit it
    whole. (Replaced test_a_resubmission_after_a_continue_round_replays_that_round,
    which pinned the replay.)"""
    ws = tmp_path / "ws"
    ws.mkdir()
    gen = _stub(
        "<compute>\nimport json\na = 2\n</compute>\n\n<compute continue>",
        "<compute>\nimport math\nb = a + 1\nc = 1 // 0\n</compute>\n",
        # The full corrected document: round 1's cells and the fixed round 2.
        "<compute>\nimport json\na = 2\n</compute>\n\n"
        "<compute>\nimport math\nb = a + 1\n</compute>\n\nB={b}.\n",
        "never asked",
    )
    out = await agent.run_literate_agent("x", base_dir=str(ws), generate=gen, policy_loader=loader,
                                         max_iterations=4)
    assert len(gen.calls) == 3
    retry = gen.calls[2][0]
    assert "a = 2" in retry and "c = 1 // 0" in retry  # the whole document so far
    assert "Rebinding" not in out
    assert "B=3." in out


@pytest.mark.asyncio
async def test_a_resubmitted_diff_that_is_already_applied_says_so(agent, loader, tmp_path):
    """A resubmission whose @diff already ran in the failed round: the file holds
    the post-image, so the cell fails with "diff already applied", nothing written."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.txt").write_text("a\nb\n")
    diff = "```lackpy @diff(c.txt)\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n```\n\n"
    gen = _stub(diff + "<compute>\nz = 1 // 0\n</compute>\n",
                diff + "<compute>\nz = 1\n</compute>\n", "Done.\n")
    await agent.run_literate_agent("x", base_dir=str(ws), generate=gen, policy_loader=loader,
                                   max_iterations=3)
    assert (ws / "c.txt").read_text() == "a\nB\n"
    assert len(gen.calls) == 3
    assert "diff already applied" in gen.calls[2][0]  # the error the model is shown
