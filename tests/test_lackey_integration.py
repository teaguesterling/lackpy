"""Integration test: full Lackey roundtrip."""

import pytest
from pathlib import Path
from lackpy.service import LackpyService


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("hello world")
    config_dir = tmp_path / ".lackpy"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('[inference]\norder = ["templates", "rules"]\n\n[kit]\ndefault = "debug"\n')
    return tmp_path


@pytest.fixture
def service(workspace):
    return LackpyService(workspace=workspace)


class TestFullRoundtrip:
    @pytest.mark.asyncio
    async def test_generate_create_parse_run(self, service, workspace):
        # 1. Generate a program via rules
        gen = await service.generate("read file hello.txt", kit=["read_file"])
        assert "read_file(" in gen.program

        # 2. Create a Lackey file from it
        path = await service.create_lackey(
            program=gen.program,
            name="ReadHello",
            tools=["read_file"],
            params={"target": {"type": "str", "default": "hello.txt"}},
            returns="str",
            creation_log=[
                {"role": "user", "content": "read file hello.txt"},
                {"role": "assistant", "content": gen.program, "accepted": True},
            ],
            output_dir=workspace,
        )
        assert path.exists()

        # 3. Parse it back
        info = service.parse_lackey(path)
        assert info.class_name == "ReadHello"
        assert "read_file" in info.tools
        assert info.has_creation_log is True

        # 4. Run it
        result = await service.run_lackey(path)
        assert result["success"]
