"""Tests for Lackey integration in the service layer."""

import pytest
from pathlib import Path
from lackpy.service import KIT_REMOVED, LackpyService


SAMPLE_LACKEY = '''
from lackpy.lackey import Lackey, Tool

class ReadFile(Lackey):
    """Read a file and return its contents."""

    path: str = "test.txt"

    returns: str

    def run(self) -> str:
        from pathlib import Path
        content = Path(self.path).read_text()
        return content
'''

# Retired kit model: SAMPLE_LACKEY declared `read_file = Tool()` and called self.read_file.
KIT_LACKEY = '''
from lackpy.lackey import Lackey, Tool

class ReadFile(Lackey):
    read_file = Tool()

    path: str = "test.txt"

    def run(self) -> str:
        return self.read_file(self.path)
'''

REQUIRED_LACKEY = '''
from lackpy.lackey import Lackey

class Greet(Lackey):
    name: str

    def run(self) -> str:
        return "hi " + self.name
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
        assert info.tools == []
        assert "path" in info.params
        assert info.returns == "str"
        assert "Path(path).read_text()" in info.run_body


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
        assert result["success"], result["error"]
        assert result["output"] == "other content"

    @pytest.mark.asyncio
    async def test_run_lackey_params_reach_the_body_through_a_literal(self, service, lackey_file):
        # Parked item 4: the body reads `path`; run_lackey prepends the params
        # literal and one binding per param, so the run is plain Python.
        result = await service.run_lackey(lackey_file)
        assert result["program"].startswith('params = {\'path\': \'test.txt\'}\npath = params[\'path\']\n')

    @pytest.mark.asyncio
    async def test_run_lackey_rejects_a_param_of_the_wrong_type(self, service, lackey_file):
        result = await service.run_lackey(lackey_file, params={"path": 3})
        assert not result["success"]
        assert "'path'" in result["error"]

    @pytest.mark.asyncio
    async def test_run_lackey_rejects_an_unknown_param(self, service, lackey_file):
        result = await service.run_lackey(lackey_file, params={"other": "x"})
        assert not result["success"]
        assert "'other'" in result["error"]

    @pytest.mark.asyncio
    async def test_required_param_must_be_passed(self, service, workspace):
        path = workspace / "greet.py"
        path.write_text(REQUIRED_LACKEY)
        missing = await service.run_lackey(path)
        assert not missing["success"]
        assert "'name'" in missing["error"]
        passed = await service.run_lackey(path, params={"name": "bo"})
        assert passed["success"], passed["error"]
        assert passed["output"] == "hi bo"

    @pytest.mark.asyncio
    async def test_lackey_declaring_tools_is_an_error(self, service, workspace):
        path = workspace / "kit_lackey.py"
        path.write_text(KIT_LACKEY)
        result = await service.run_lackey(path)
        assert not result["success"]
        assert result["error"] == KIT_REMOVED


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
