"""Tests for Lackey integration in the service layer."""

import pytest
from pathlib import Path
from lackpy.service import LackpyService


SAMPLE_LACKEY = '''
from lackpy.lackey import Lackey, Tool

class ReadFile(Lackey):
    """Read a file and return its contents."""

    read_file = Tool()

    path: str = "test.txt"

    returns: str

    def run(self) -> str:
        content = self.read_file(self.path)
        return content
'''


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test.txt").write_text("hello from lackey")
    return tmp_path


@pytest.fixture
def service(workspace):
    return LackpyService(workspace=workspace)


@pytest.fixture
def lackey_file(workspace):
    path = workspace / "read_file.py"
    path.write_text(SAMPLE_LACKEY)
    return path


class TestParseLackey:
    def test_parse_lackey(self, service, lackey_file):
        info = service.parse_lackey(lackey_file)
        assert info.class_name == "ReadFile"
        assert "read_file" in info.tools
        assert "path" in info.params
        assert info.returns == "str"
        assert "read_file(path)" in info.run_body


class TestRunLackey:
    @pytest.mark.asyncio
    async def test_run_lackey_default_params(self, service, lackey_file):
        result = await service.run_lackey(lackey_file)
        assert result["success"]
        assert result["output"] == "hello from lackey"

    @pytest.mark.asyncio
    async def test_run_lackey_override_params(self, service, lackey_file, workspace):
        (workspace / "other.txt").write_text("other content")
        result = await service.run_lackey(lackey_file, params={"path": "other.txt"})
        assert result["success"]
        assert result["output"] == "other content"


class TestCreateLackey:
    @pytest.mark.asyncio
    async def test_create_lackey(self, service, workspace):
        path = await service.create_lackey(
            program="content = read('test.txt')\ncontent",
            name="ReadTest",
            tools=["read"],
            params={"target": {"type": "str", "default": "test.txt"}},
            returns="str",
            output_dir=workspace,
        )
        assert path.exists()
        assert "class ReadTest(Lackey):" in path.read_text()

        info = service.parse_lackey(path)
        assert info.class_name == "ReadTest"
        assert "read" in info.tools
