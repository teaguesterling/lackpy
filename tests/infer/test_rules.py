"""Tests for the rules inference provider."""

import pytest

from lackpy.infer.providers.rules import RulesProvider


@pytest.fixture
def provider():
    return RulesProvider()


class TestRuleMatching:
    @pytest.mark.asyncio
    async def test_read_file_rule(self, provider):
        result = await provider.generate("read file src/main.py", namespace_desc="  read_file(path) -> str: Read file")
        assert result is not None
        assert "read_file(" in result
        assert "src/main.py" in result

    @pytest.mark.asyncio
    async def test_find_definitions_rule(self, provider):
        result = await provider.generate("find definitions of MyClass", namespace_desc="  find_definitions(name) -> list: Find defs")
        assert result is not None
        assert "find_definitions(" in result
        assert "MyClass" in result

    @pytest.mark.asyncio
    async def test_glob_rule(self, provider):
        result = await provider.generate("find all python files", namespace_desc="  find_files(pattern) -> list: Find files")
        assert result is not None
        assert "find_files(" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, provider):
        result = await provider.generate("optimize the quantum flux capacitor", namespace_desc="")
        assert result is None

class TestAvailability:
    def test_always_available(self, provider):
        assert provider.available()
