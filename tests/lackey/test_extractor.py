"""Tests for extracting and rewriting run() bodies."""

from lackpy.lackey.extractor import extract_run_source, rewrite_self_to_plain


class TestExtractRunSource:
    def test_extracts_run_body(self):
        source = '''
class CountLines(Lackey):
    read = Tool()
    pattern: str = "**/*.py"

    def run(self) -> int:
        files = self.glob(self.pattern)
        for f in files:
            content = self.read(f)
            print(f)
        return len(files)
'''
        body = extract_run_source(source, "CountLines")
        assert "self.glob" in body
        assert "self.read" in body
        assert "def run" not in body

    def test_raises_if_no_run(self):
        source = '''
class NoRun(Lackey):
    read = Tool()
'''
        import pytest
        with pytest.raises(ValueError, match="No run\\(\\) method"):
            extract_run_source(source, "NoRun")

    def test_raises_if_class_not_found(self):
        source = "x = 1"
        import pytest
        with pytest.raises(ValueError, match="not found"):
            extract_run_source(source, "Missing")


class TestRewriteSelfToPlain:
    def test_rewrites_self_tool_calls(self):
        code = "files = self.glob(self.pattern)\nself.read(f)"
        result = rewrite_self_to_plain(code)
        assert "glob(pattern)" in result
        assert "read(f)" in result
        assert "self." not in result

    def test_rewrites_self_attribute_access(self):
        code = "x = self.pattern"
        result = rewrite_self_to_plain(code)
        assert result.strip() == "x = pattern"

    def test_leaves_non_self_alone(self):
        code = "x = len(files)\nprint(x)"
        result = rewrite_self_to_plain(code)
        assert result == code

    def test_handles_nested_self(self):
        code = "print(f\"{self.name}: {len(self.read(self.path))}\")"
        result = rewrite_self_to_plain(code)
        assert "self." not in result
        assert "name" in result
        assert "read(path)" in result
