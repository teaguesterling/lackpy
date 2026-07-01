"""Tests for the unified service layer."""

import pytest

from lackpy.service import LackpyService
from lackpy.profiles import Profile
from lackpy.tools.toolbox import ToolSpec, ArgSpec


@pytest.fixture
def service(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    svc.toolbox.register_tool(ToolSpec(
        name="read_file", provider="builtin",
        description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    ))
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    return svc


class TestValidate:
    def test_valid_program(self, service):
        result = service.validate("x = read_file('test.txt')\nlen(x)", profile=["read_file"])
        assert result.valid
        assert "read_file" in result.calls

    def test_invalid_program(self, service):
        result = service.validate("import os", profile=["read_file"])
        assert not result.valid


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_rules(self, service):
        result = await service.generate("read file test.txt", profile=["read_file"])
        assert result.program is not None
        assert "read_file(" in result.program

    @pytest.mark.asyncio
    async def test_generate_no_match(self, service):
        with pytest.raises(RuntimeError):
            await service.generate("do something impossibly vague", profile=["read_file"])


class TestRunProgram:
    @pytest.mark.asyncio
    async def test_run_valid_program(self, service):
        result = await service.run_program("x = read_file('test.txt')\nlen(x)", profile=["read_file"])
        assert result.success
        assert result.output == 11

    @pytest.mark.asyncio
    async def test_run_invalid_program(self, service):
        result = await service.run_program("import os", profile=["read_file"])
        assert not result.success

    @pytest.mark.asyncio
    async def test_run_program_print_recovered_via_effective_output(self, service):
        # A program that prints its answer (instead of a bare last expression)
        # must not silently drop the value: the typed output is None, but the
        # printed text is captured and surfaced via effective_output.
        result = await service.run_program("print(read_file('test.txt'))", profile=["read_file"])
        assert result.success
        assert result.output is None
        assert result.stdout == "hello world\n"
        assert result.effective_output == "hello world"


class TestDelegate:
    @pytest.mark.asyncio
    async def test_delegate_simple(self, service):
        result = await service.delegate("read file test.txt", profile=["read_file"])
        assert result["success"]
        assert "read_file" in result["program"]

    @pytest.mark.asyncio
    async def test_delegate_result_carries_stdout(self, service):
        # The delegate contract surfaces captured stdout so a printed answer is
        # never lost, even when the typed output is populated.
        result = await service.delegate("read file test.txt", profile=["read_file"])
        assert "stdout" in result

    @pytest.mark.asyncio
    async def test_delegate_with_params(self, service):
        result = await service.delegate("read file test.txt", profile=["read_file"], params={"prefix": "hello"})
        assert result["success"]


class TestKitInfo:
    def test_kit_info_from_list(self, service):
        info = service.profile_info(["read_file"])
        assert "read_file" in info["tools"]
        assert info["grade"]["w"] == 1


class TestGetConfig:
    def test_returns_dict(self, service):
        config = service.get_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self, service):
        config = service.get_config()
        assert "inference_order" in config
        assert "profile_default" in config
        assert "sandbox_enabled" in config
        assert "config_dir" in config

    def test_config_dir_is_string(self, service):
        config = service.get_config()
        assert isinstance(config["config_dir"], str)


class TestProviderList:
    def test_returns_list(self, service):
        providers = service.provider_list()
        assert isinstance(providers, list)

    def test_providers_have_required_keys(self, service):
        providers = service.provider_list()
        # At minimum templates and rules are always present
        assert len(providers) >= 2
        for p in providers:
            assert "name" in p
            assert "plugin" in p
            assert "available" in p

    def test_templates_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "templates" in names

    def test_rules_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "rules" in names


class TestLanguageSpec:
    def test_returns_dict(self, service):
        spec = service.language_spec()
        assert isinstance(spec, dict)

    def test_has_spec_keys(self, service):
        spec = service.language_spec()
        assert "allowed_nodes" in spec
        assert "allowed_builtins" in spec


class TestLiterateExecutionAxis:
    """Profile `execution == "literate"` routes run_program through the
    LiterateInterpreter, so the effect-ceiling gate (defaulting to the granted
    toolset's grade) and the write journal enforce end-to-end via the service.
    """

    @pytest.mark.asyncio
    async def test_write_doc_refused_under_a_read_only_literate_profile(self, service):
        # read_file grants only a w=1 toolset -> ceiling w=1 -> a @write doc (w=3)
        # is refused before any cell runs, and the file is never created.
        doc = "```lackpy @write(out.txt)\nhello\n```"
        profile = Profile(tools=["read_file"], execution="literate")
        result = await service.run_program(doc, profile=profile)
        assert not result.success
        assert "effect ceiling exceeded" in (result.error or "")
        assert not (service._workspace / "out.txt").exists()

    @pytest.mark.asyncio
    async def test_within_ceiling_doc_executes_and_renders(self, service):
        # A pure doc (w=0) is within the read ceiling -> it runs and renders.
        doc = "```lackpy\nx = 2 + 2\n```\n\nResult: {x}"
        profile = Profile(tools=["read_file"], execution="literate")
        result = await service.run_program(doc, profile=profile)
        assert result.success
        assert "Result: 4" in result.stdout

    @pytest.mark.asyncio
    async def test_write_allowed_when_toolset_grade_permits(self, service):
        # Granting a w=3 tool (edit_file) raises the ceiling to w=3, so the @write
        # doc is allowed and the literate write builtin creates the file. edit_file
        # is deliberately NOT write_file, so it does not shadow that builtin.
        service.toolbox.register_tool(ToolSpec(
            name="edit_file", provider="builtin", description="edit a file",
            args=[ArgSpec(name="path", type="str", description="File path")],
            returns="str", grade_w=3, effects_ceiling=3,
        ))
        doc = "```lackpy @write(out.txt)\nkept\n```"
        profile = Profile(tools=["read_file", "edit_file"], execution="literate")
        result = await service.run_program(doc, profile=profile)
        assert result.success, result.error
        assert (service._workspace / "out.txt").read_text() == "kept"

    @pytest.mark.asyncio
    async def test_one_shot_axis_unaffected(self, service):
        # The default execution axis still runs restricted Python through _execute.
        result = await service.run_program(
            "x = read_file('test.txt')\nlen(x)", profile=["read_file"])
        assert result.success

    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_gate_refuses(self, service):
        # delegate routes through the same axis seam: a write doc under a read-only
        # literate profile is refused by the gate (via _program_override, no LLM).
        ro = Profile(tools=["read_file"], execution="literate")
        result = await service.delegate(
            "x", profile=ro, _program_override="```lackpy @write(o.txt)\nhi\n```")
        assert not result["success"]
        assert "effect ceiling exceeded" in (result["error"] or "")
        assert not (service._workspace / "o.txt").exists()

    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_renders(self, service):
        ro = Profile(tools=["read_file"], execution="literate")
        result = await service.delegate(
            "x", profile=ro, _program_override="```lackpy\nx = 2 + 2\n```\n\nResult: {x}")
        assert result["success"]
        assert "Result: 4" in result["stdout"]


class TestLiterateGenerationRouting:
    """generate() routes an execution=="literate" profile to the literate
    generation path (own hint + parse validation), not the Python dispatcher.
    """

    @pytest.mark.asyncio
    async def test_literate_profile_uses_the_literate_generation_path(self, service, monkeypatch):
        from lackpy.infer.dispatch import GenerationResult
        seen = {}

        async def fake_gen_literate(intent, providers, resolved, params_desc, rules):
            seen["intent"] = intent
            return GenerationResult(program="Hello {x}", provider_name="fake", generation_time_ms=0.0)

        monkeypatch.setattr(service, "_generate_literate", fake_gen_literate)
        result = await service.generate(
            "greet", profile=Profile(tools=["read_file"], execution="literate"))
        assert seen.get("intent") == "greet"
        assert result.program == "Hello {x}"

    @pytest.mark.asyncio
    async def test_one_shot_profile_does_not_use_the_literate_path(self, service, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("_generate_literate must not run for a one-shot profile")

        monkeypatch.setattr(service, "_generate_literate", boom)
        result = await service.generate("read file test.txt", profile=["read_file"])
        assert "read_file(" in result.program


class TestStripLiterateWrapper:
    def test_strips_leading_preamble(self):
        from lackpy.service import _strip_literate_wrapper
        out = _strip_literate_wrapper("Here is the document:\n```lackpy\nx=1\n```")
        assert out == "```lackpy\nx=1\n```"

    def test_strips_outer_markdown_wrapper(self):
        from lackpy.service import _strip_literate_wrapper
        assert _strip_literate_wrapper("```markdown\nHello {x}\n```") == "Hello {x}"

    def test_preserves_inner_lackpy_fences(self):
        from lackpy.service import _strip_literate_wrapper
        doc = "```lackpy @write(o.txt)\nv=1\n```"
        assert _strip_literate_wrapper(doc) == doc  # must NOT unwrap the code block

    def test_empty_is_empty(self):
        from lackpy.service import _strip_literate_wrapper
        assert _strip_literate_wrapper("   ") == ""
