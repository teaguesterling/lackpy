"""Tests for the Anthropic inference provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lackpy.infer.providers.anthropic import AnthropicProvider


@pytest.fixture
def provider():
    return AnthropicProvider(model="claude-haiku-4-5-20251001")


class TestGeneration:
    @pytest.mark.asyncio
    async def test_generate_returns_content(self, provider):
        mock_block = MagicMock()
        mock_block.text = "x = read('test.py')\nlen(x)"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        with patch.object(provider, "available", return_value=True), \
             patch.object(provider, "_create_message", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate("read test.py", namespace_desc="  read(path) -> str: Read file")
            assert result is not None
            assert "read(" in result
