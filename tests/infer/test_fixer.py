"""Tests for the fixer prompt builder."""

from lackpy.infer.fixer import build_fixer_prompt, build_fixer_messages


class TestBuildFixerPrompt:
    def test_contains_namespace(self):
        prompt = build_fixer_prompt("  read(path) -> str: Read file contents")
        assert "read(path)" in prompt

    def test_has_fixer_framing(self):
        prompt = build_fixer_prompt("")
        assert "fixing" in prompt.lower() or "fix" in prompt.lower()
        assert "errors" in prompt.lower()

    def test_not_jupyter_framing(self):
        prompt = build_fixer_prompt("")
        assert "Jupyter" not in prompt
        assert "notebook" not in prompt.lower()

    def test_contains_builtins(self):
        prompt = build_fixer_prompt("")
        assert "len" in prompt
        assert "print" in prompt

    def test_output_only_instruction(self):
        prompt = build_fixer_prompt("")
        assert "ONLY" in prompt


class TestBuildFixerMessages:
    def test_includes_broken_code(self):
        messages = build_fixer_messages(
            intent="list files",
            broken_program="import os\nos.listdir('.')",
            errors="NameError: os not defined",
            namespace_desc="  ls(path) -> list: List files",
        )
        user_msg = messages[1]["content"]
        assert "import os" in user_msg
        assert "os.listdir" in user_msg

    def test_includes_errors(self):
        messages = build_fixer_messages(
            intent="list files",
            broken_program="bad code",
            errors="SyntaxError: invalid syntax",
            namespace_desc="",
        )
        user_msg = messages[1]["content"]
        assert "SyntaxError" in user_msg

    def test_includes_intent(self):
        messages = build_fixer_messages(
            intent="find large files",
            broken_program="x = 1",
            errors="",
            namespace_desc="",
        )
        user_msg = messages[1]["content"]
        assert "find large files" in user_msg

    def test_correct_message_structure(self):
        messages = build_fixer_messages(
            intent="do something",
            broken_program="x = 1",
            errors="no errors",
            namespace_desc="read(path) -> str",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)

    def test_system_has_fixer_framing(self):
        messages = build_fixer_messages(
            intent="x",
            broken_program="x",
            errors="x",
            namespace_desc="read(path) -> str",
        )
        system_content = messages[0]["content"]
        assert "fixing" in system_content.lower() or "fix" in system_content.lower()
        assert "Jupyter" not in system_content
