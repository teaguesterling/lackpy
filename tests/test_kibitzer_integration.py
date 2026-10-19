"""Tests for Kibitzer integration in the service layer."""

import pytest
from pathlib import Path

from lackpy.service import LackpyService, _HAS_KIBITZER

READ = "from pathlib import Path\nPath('test.txt').read_text()"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test.txt").write_text("hello")
    config_dir = tmp_path / ".lackpy"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[inference]\norder = ["templates", "rules"]\n\n[kit]\ndefault = "debug"\n'
    )
    # Create minimal .kibitzer config so the session can load
    kib_dir = tmp_path / ".kibitzer"
    kib_dir.mkdir()
    (kib_dir / "config.toml").write_text(
        '[mode]\ncurrent = "free"\n'
    )
    return tmp_path


@pytest.fixture
def service(workspace):
    return LackpyService(workspace=workspace)


@pytest.mark.skipif(not _HAS_KIBITZER, reason="kibitzer not installed")
class TestKibitzerAvailable:
    def test_kibitzer_session_initialized(self, service):
        assert service._kibitzer is not None

    @pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; kibitzer integration migration pending", strict=True)
    def test_tools_registered_with_kibitzer(self, service):
        # Our builtin tools should be registered
        registered = service._kibitzer.registered_tools
        assert "read_file" in registered
        assert "find_files" in registered

    @pytest.mark.asyncio
    async def test_delegate_with_kibitzer(self, service):
        # Retired kit model: generated a read_file program from profile=["read_file"].
        result = await service.delegate("read file test.txt", _program_override=READ)
        assert result["success"]
        # Kibitzer should have tracked the read call
        assert result["output"] == "hello"

    @pytest.mark.asyncio
    async def test_delegate_returns_kibitzer_suggestions(self, service):
        result = await service.delegate("read file test.txt", _program_override=READ)
        # suggestions key may or may not be present depending on kibitzer state
        # but the delegate should not crash
        assert result["success"]


class TestKibitzerGracefulDegradation:
    def test_service_works_without_kibitzer_config(self, tmp_path):
        """Service should work even if .kibitzer/ doesn't exist."""
        (tmp_path / "test.txt").write_text("hello")
        svc = LackpyService(workspace=tmp_path)
        # Should not crash — kibitzer init may fail gracefully
        # Retired kit model: validated a read_file call under profile=["read_file"].
        result = svc.validate(READ)
        assert result.valid
