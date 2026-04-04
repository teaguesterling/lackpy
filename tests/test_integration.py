"""Integration tests: full pipeline from intent to trace."""

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec


@pytest.fixture
def workspace(tmp_path):
    config_dir = tmp_path / ".lackpy"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('[inference]\norder = ["templates", "rules"]\n\n[kit]\ndefault = "debug"\n')
    kits_dir = config_dir / "kits"
    kits_dir.mkdir()
    (kits_dir / "debug.kit").write_text("---\nname: debug\ndescription: Read-only\n---\nread_file\nfind_files\n")
    templates_dir = config_dir / "templates"
    templates_dir.mkdir()
    (templates_dir / "read-file.tmpl").write_text(
        '---\nname: read-file\npattern: "read (the )?file {path}"\nsuccess_count: 5\nfail_count: 0\n---\n'
        "content = read_file('{path}')\ncontent\n"
    )
    (tmp_path / "hello.txt").write_text("hello world")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    return tmp_path


@pytest.fixture
def service(workspace):
    svc = LackpyService(workspace=workspace)
    for name, desc, grade in [("read_file", "Read file contents", 1), ("find_files", "Find files matching pattern", 1)]:
        svc.toolbox.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path" if name == "read_file" else "pattern", type="str")],
            returns="str" if name == "read_file" else "list[str]",
            grade_w=grade, effects_ceiling=grade,
        ))
    return svc


class TestTemplateDelegate:
    @pytest.mark.asyncio
    async def test_read_file_via_template(self, service):
        result = await service.delegate("read the file hello.txt", kit="debug")
        assert result["success"]
        assert result["generation_tier"] == "templates"
        assert result["output"] == "hello world"
        assert any(e["tool"] == "read_file" for e in result["trace"])


class TestRulesDelegate:
    @pytest.mark.asyncio
    async def test_read_file_via_rules(self, service):
        result = await service.delegate("read file hello.txt", kit=["read_file"])
        assert result["success"]
        assert result["output"] == "hello world"


class TestRunProgram:
    @pytest.mark.asyncio
    async def test_run_with_glob(self, service):
        result = await service.run_program("files = find_files('src/*.py')\nlen(files)", kit=["find_files"])
        assert result.success
        assert result.output == 1


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_template(self, service):
        result = await service.create(
            "content = read_file('{path}')\nlen(content)",
            kit=["read_file"], name="file-length", pattern="how long is {path}",
        )
        assert result["success"]
        assert Path(result["path"]).exists()


class TestKitManagement:
    def test_kit_list(self, service):
        kits = service.kit_list()
        assert any(k["name"] == "debug" for k in kits)

    def test_kit_create_and_info(self, service):
        service.kit_create("readonly", ["read_file", "find_files"], "Read-only tools")
        info = service.kit_info("readonly")
        assert "read_file" in info["tools"]
        assert info["grade"]["w"] == 1


class TestParamsIntegration:
    @pytest.mark.asyncio
    async def test_run_with_params(self, service):
        result = await service.run_program("len(content)", kit=["read_file"], params={"content": "test string"})
        assert result.success
        assert result.output == 11
