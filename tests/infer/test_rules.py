"""Tests for the rules inference provider.

stepy100 Task 8: the rules emit Path programs. The kit-era tests asserted
read_file(, find_definitions( and find_files( calls; the kit is gone
(Plan 2, Task 7), so the definitions rule (no Path form) is retired too.
"""

import subprocess
import sys

import pytest

from lackpy.infer.providers.rules import RulesProvider
from lackpy.lang.validator import validate

KIT_NAMES = ["read_file", "find_files", "find_definitions", "find_callers"]


@pytest.fixture
def provider():
    return RulesProvider()


def _python3(program, cwd):
    """Run under plain CPython, as the plain-Python invariant says."""
    return subprocess.run([sys.executable, "-I", "-c", program], cwd=cwd,
                          capture_output=True, text=True, timeout=30)


class TestRuleMatching:
    @pytest.mark.asyncio
    async def test_read_file_rule(self, provider, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')\n")
        result = await provider.generate("read file src/main.py", namespace_desc="")
        assert result is not None
        assert "Path(" in result
        assert validate(result, allowed_names=set()).valid, validate(result).errors
        assert _python3(result, tmp_path).stdout == "print('hi')\n"

    @pytest.mark.asyncio
    async def test_read_file_rule_quotes_the_path(self, provider):
        result = await provider.generate("read file a'b.txt", namespace_desc="")
        assert result is not None
        assert validate(result, allowed_names=set()).valid

    @pytest.mark.asyncio
    async def test_glob_rule(self, provider, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await provider.generate("find all python files", namespace_desc="")
        assert result is not None
        assert validate(result, allowed_names=set()).valid
        assert _python3(result, tmp_path).stdout.split() == ["a.py", "pkg/b.py"]

    @pytest.mark.asyncio
    async def test_explicit_glob_rule(self, provider, tmp_path):
        (tmp_path / "a.md").write_text("")
        result = await provider.generate("glob *.md", namespace_desc="")
        assert result is not None
        assert validate(result, allowed_names=set()).valid
        assert _python3(result, tmp_path).stdout.split() == ["a.md"]

    @pytest.mark.asyncio
    async def test_rules_need_no_namespace(self, provider):
        # The kit-era rules fired only when the namespace named the tool.
        assert await provider.generate("read file x.txt", namespace_desc="") is not None

    @pytest.mark.asyncio
    async def test_no_rule_emits_a_kit_call(self, provider):
        intents = ["read file x.txt", "find all python files", "glob *.md",
                   "find definitions of MyClass", "find callers of f"]
        for intent in intents:
            result = await provider.generate(intent, namespace_desc="")
            for name in KIT_NAMES:
                assert result is None or name not in result, (intent, result)

    @pytest.mark.asyncio
    async def test_definitions_have_no_path_form(self, provider):
        assert await provider.generate("find definitions of MyClass", namespace_desc="") is None

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, provider):
        result = await provider.generate("optimize the quantum flux capacitor", namespace_desc="")
        assert result is None


class TestAvailability:
    def test_always_available(self, provider):
        assert provider.available()
