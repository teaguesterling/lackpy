"""Tests for the template inference provider.

stepy100 Task 8: the fixture templates are Path programs. The kit-era fixtures
held find_callers(...) and read_file(...) calls; the kit is gone (Plan 2, Task 7).
"""

import pytest

from lackpy.infer.providers.templates import TemplatesProvider
from lackpy.lang.validator import validate


@pytest.fixture
def templates_dir(tmp_path):
    t = tmp_path / "count-lines.tmpl"
    t.write_text(
        '---\nname: count-lines\n'
        'pattern: "count (the )?lines (in|of) {path}"\n'
        'success_count: 10\nfail_count: 0\n---\n'
        'from pathlib import Path\n\nprint(len(Path("{path}").read_text().splitlines()))\n'
    )
    t2 = tmp_path / "read-file.tmpl"
    t2.write_text(
        '---\nname: read-file\n'
        'pattern: "read (the )?file {path}"\n'
        'success_count: 5\nfail_count: 1\n---\n'
        'from pathlib import Path\n\ntext = Path("{path}").read_text()\nprint(text, end="")\n'
    )
    return tmp_path


@pytest.fixture
def provider(templates_dir):
    return TemplatesProvider(templates_dir=templates_dir)


class TestTemplateMatching:
    @pytest.mark.asyncio
    async def test_matches_count_lines(self, provider):
        result = await provider.generate("count the lines in notes.md", namespace_desc="")
        assert result is not None
        assert 'Path("notes.md")' in result

    @pytest.mark.asyncio
    async def test_matches_read_file(self, provider):
        result = await provider.generate("read the file src/main.py", namespace_desc="")
        assert result is not None
        assert 'Path("src/main.py").read_text()' in result
        assert validate(result, allowed_names=set()).valid

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
