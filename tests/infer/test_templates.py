"""Tests for the template inference provider."""

import pytest

from lackpy.infer.providers.templates import TemplatesProvider


@pytest.fixture
def templates_dir(tmp_path):
    t = tmp_path / "find-callers.tmpl"
    t.write_text(
        '---\nname: find-callers\n'
        'pattern: "find (all )?(callers|usages|references) of {name}"\n'
        'success_count: 10\nfail_count: 0\n---\n'
        'results = find_callers("{name}")\nresults\n'
    )
    t2 = tmp_path / "read-file.tmpl"
    t2.write_text(
        '---\nname: read-file\n'
        'pattern: "read (the )?file {path}"\n'
        'success_count: 5\nfail_count: 1\n---\n'
        "content = read('{path}')\ncontent\n"
    )
    return tmp_path


@pytest.fixture
def provider(templates_dir):
    return TemplatesProvider(templates_dir=templates_dir)


class TestTemplateMatching:
    @pytest.mark.asyncio
    async def test_matches_find_callers(self, provider):
        result = await provider.generate("find all callers of validate_token", namespace_desc="")
        assert result is not None
        assert "find_callers" in result
        assert "validate_token" in result

    @pytest.mark.asyncio
    async def test_matches_read_file(self, provider):
        result = await provider.generate("read the file src/main.py", namespace_desc="")
        assert result is not None
        assert "read" in result
        assert "src/main.py" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, provider):
        result = await provider.generate("do something completely unrelated", namespace_desc="")
        assert result is None


class TestAvailability:
    def test_available_with_templates(self, provider):
        assert provider.available()

    def test_not_available_without_dir(self, tmp_path):
        p = TemplatesProvider(templates_dir=tmp_path / "nonexistent")
        assert not p.available()
