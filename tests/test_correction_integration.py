"""Integration test: correction chain in delegate results."""

import pytest
from lackpy.service import LackpyService


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test.txt").write_text("hello")
    config_dir = tmp_path / ".lackpy"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[inference]\norder = ["templates", "rules"]\n\n[kit]\ndefault = "debug"\n'
    )
    return tmp_path


@pytest.fixture
def service(workspace):
    return LackpyService(workspace=workspace)


class TestCorrectionInDelegate:
    @pytest.mark.asyncio
    async def test_delegate_returns_correction_info(self, service):
        result = await service.delegate("read file test.txt", kit=["read"])
        assert result["success"]
        assert "correction_strategy" in result
        assert "correction_attempts" in result

    @pytest.mark.asyncio
    async def test_correction_fields_have_correct_types(self, service):
        result = await service.delegate("read file test.txt", kit=["read"])
        assert result["correction_strategy"] is None or isinstance(result["correction_strategy"], str)
        assert isinstance(result["correction_attempts"], int)

    @pytest.mark.asyncio
    async def test_program_override_has_correction_defaults(self, service):
        """_program_override path should have None/0 correction fields."""
        result = await service.delegate(
            "read file test.txt", kit=["read"], _program_override='read("test.txt")'
        )
        assert result["correction_strategy"] is None
        assert result["correction_attempts"] == 0
