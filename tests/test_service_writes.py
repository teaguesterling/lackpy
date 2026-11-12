"""``run_program(..., declared_writes=...)`` (spec E7, Task 7).

The declared writes narrow the settings policy on both execution axes: a
one-shot program and a literate document. ``None`` keeps today's behaviour
for ``lackpy``, ``lackpyctl`` and the MCP tools. The policy loader is a stub
over a temporary home, so no test reads the real ``~/.claude``.
"""

import pytest

from lackpy.profiles import Profile
from lackpy.run.paths import load_policy
from lackpy.service import LackpyService

_LIT = Profile(execution="literate")


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


@pytest.fixture
def service(tmp_path, ws):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return LackpyService(workspace=ws,
                         path_policy_loader=lambda w: load_policy(w, home=home, managed=None))


def _write(path):
    # mkdir as well: the declared writes allow the parent of a declared path.
    return (f"from pathlib import Path\np = Path({path!r})\n"
            f"p.parent.mkdir(parents=True, exist_ok=True)\np.write_text('x')")


def _program(path):
    return _write(path)


def _document(path):
    return f"```lackpy\n{_write(path)}\n```\n"


_AXES = [(None, _program), (_LIT, _document)]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,text", _AXES)
async def test_a_declared_write_succeeds(service, ws, profile, text):
    result = await service.run_program(text("out/a.txt"), profile=profile,
                                       declared_writes=["out/*.txt"])
    assert result.success, result.error
    assert (ws / "out" / "a.txt").read_text() == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,text", _AXES)
async def test_an_undeclared_write_is_refused_and_names_the_globs(service, ws, profile, text):
    result = await service.run_program(text("other.txt"), profile=profile,
                                       declared_writes=["out/*.txt"])
    assert not result.success
    assert result.error_phase == "refused", result.error
    assert "other.txt" in result.error and "out/*.txt" in result.error
    assert not (ws / "other.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,text", _AXES)
async def test_no_declared_writes_refuses_every_write(service, ws, profile, text):
    result = await service.run_program(text("a.txt"), profile=profile, declared_writes=[])
    assert not result.success
    assert "no declared writes" in result.error
    assert not (ws / "a.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,text", _AXES)
async def test_none_keeps_todays_behaviour(service, ws, profile, text):
    result = await service.run_program(text("a.txt"), profile=profile)
    assert result.success, result.error
    assert (ws / "a.txt").read_text() == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,text", _AXES)
async def test_a_glob_out_of_the_workspace_runs_nothing(service, ws, tmp_path, profile, text):
    # A {name} filled in as "../x" cannot move a glob out (Review Focus 3).
    result = await service.run_program(text("../escaped.txt"), profile=profile,
                                       declared_writes=["../*.txt"])
    assert not result.success
    assert "declared writes glob" in result.error
    assert not (tmp_path / "escaped.txt").exists()
