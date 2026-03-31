"""Test pluckit mock kit registration and basic execution."""

from __future__ import annotations

import pytest
from pathlib import Path

from lackpy.kit.providers.mock import MockProvider
from lackpy.kit.providers.pluckit_tools import PLUCKIT_TOOLS
from lackpy.kit.providers.pluckit_registration import register_pluckit_tools
from lackpy.kit.toolbox import Toolbox
from lackpy.kit.registry import resolve_kit
from lackpy.lang.validator import validate
from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def toolbox():
    tb = Toolbox()
    register_pluckit_tools(tb)
    return tb


class TestPluckitRegistration:
    def test_mock_provider_registered(self, toolbox):
        assert toolbox.resolve("select") is not None

    def test_all_tools_registered(self, toolbox):
        for spec in PLUCKIT_TOOLS:
            assert toolbox.resolve(spec.name) is not None

    def test_tool_count(self, toolbox):
        assert len(PLUCKIT_TOOLS) > 50

    def test_resolve_pluckit_kit(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\nfind\ncallers\ntext\naddParam\nsave\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        assert "select" in kit.tools
        assert "callers" in kit.tools
        assert "addParam" in kit.tools
        assert "save" in kit.tools

    def test_kit_description_includes_tools(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\ncallers\naddParam\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        desc = kit.description
        assert "select" in desc
        assert "callers" in desc
        assert "addParam" in desc


class TestPluckitValidation:
    def test_simple_select_validates(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\ncallers\ntext\ncount\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        result = validate(
            'fns = select(".fn:exported")\ncount(fns)',
            allowed_names=set(kit.tools.keys()),
        )
        assert result.valid, result.errors

    def test_chain_with_mutation_validates(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\naddParam\nsave\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        result = validate(
            'fns = select(".fn:exported")\nresult = addParam(fns, "timeout: int = 30")\nsave(result, "feat: add timeout")',
            allowed_names=set(kit.tools.keys()),
        )
        assert result.valid, result.errors

    def test_unknown_tool_rejected(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\ncallers\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        result = validate(
            'fns = select(".fn")\ndelete_everything(fns)',
            allowed_names=set(kit.tools.keys()),
        )
        assert not result.valid


class TestPluckitExecution:
    def test_select_returns_mock_data(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\ncallers\ncount\nnames\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        runner = RestrictedRunner()
        result = runner.run(
            'fns = select(".fn#validate_token")\ncaller_fns = callers(fns)\nnames(caller_fns)',
            kit.callables,
        )
        assert result.success, result.error
        assert isinstance(result.output, list)
        assert len(result.trace.entries) == 3

    def test_mutation_chain_executes(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\naddParam\nsave\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        runner = RestrictedRunner()
        result = runner.run(
            'fns = select(".fn:exported")\nmutated = addParam(fns, "timeout: int = 30")\nsave(mutated, "feat: add timeout")',
            kit.callables,
        )
        assert result.success, result.error
        assert result.output["committed"] is True
        assert len(result.trace.entries) == 3

    def test_query_chain_executes(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\nsimilar\ncompare\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        runner = RestrictedRunner()
        result = runner.run(
            'fns = select(".fn#validate_token")\nsim = similar(fns, 0.7)\ncompare(sim)',
            kit.callables,
        )
        assert result.success, result.error
        assert "functions" in result.output

    def test_history_chain_executes(self, toolbox, tmp_path):
        kits_dir = tmp_path / "kits"
        kits_dir.mkdir()
        (kits_dir / "pluckit.kit").write_text(
            "---\nname: pluckit\n---\nselect\nhistory\nat\ndiff\n"
        )
        kit = resolve_kit("pluckit", toolbox, kits_dir=kits_dir)
        runner = RestrictedRunner()
        result = runner.run(
            'fn = select(".fn#validate_token")\nold = at(fn, "last_green_build")\ndiff(fn, old)',
            kit.callables,
        )
        assert result.success, result.error
        assert "added" in result.output
