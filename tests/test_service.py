"""Tests for the unified service layer."""

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec


@pytest.fixture
def service(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    svc.toolbox.register_tool(ToolSpec(
        name="read", provider="builtin",
        description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    ))
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    return svc


class TestValidate:
    def test_valid_program(self, service):
        result = service.validate("x = read('test.txt')\nlen(x)", kit=["read"])
        assert result.valid
        assert "read" in result.calls

    def test_invalid_program(self, service):
        result = service.validate("import os", kit=["read"])
        assert not result.valid


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_rules(self, service):
        result = await service.generate("read file test.txt", kit=["read"])
        assert result.program is not None
        assert "read(" in result.program

    @pytest.mark.asyncio
    async def test_generate_no_match(self, service):
        with pytest.raises(RuntimeError):
            await service.generate("do something impossibly vague", kit=["read"])


class TestRunProgram:
    @pytest.mark.asyncio
    async def test_run_valid_program(self, service):
        result = await service.run_program("x = read('test.txt')\nlen(x)", kit=["read"])
        assert result.success
        assert result.output == 11

    @pytest.mark.asyncio
    async def test_run_invalid_program(self, service):
        result = await service.run_program("import os", kit=["read"])
        assert not result.success


class TestDelegate:
    @pytest.mark.asyncio
    async def test_delegate_simple(self, service):
        result = await service.delegate("read file test.txt", kit=["read"])
        assert result["success"]
        assert "read" in result["program"]

    @pytest.mark.asyncio
    async def test_delegate_with_params(self, service):
        result = await service.delegate("read file test.txt", kit=["read"], params={"prefix": "hello"})
        assert result["success"]


class TestKitInfo:
    def test_kit_info_from_list(self, service):
        info = service.kit_info(["read"])
        assert "read" in info["tools"]
        assert info["grade"]["w"] == 1
