"""Tests for the unified service layer."""

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec


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
