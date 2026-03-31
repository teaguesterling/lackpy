"""Tests for the Ollama inference provider."""

import pytest
from unittest.mock import AsyncMock, patch

from lackpy.infer.providers.ollama import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider(host="http://localhost:11434", model="qwen2.5-coder:1.5b")


class TestGeneration:
    @pytest.mark.asyncio
    async def test_generate_returns_content(self, provider):
        mock_response = {"message": {"content": "x = read('test.py')\nlen(x)"}}
        with patch.object(provider, "available", return_value=True), \
             patch.object(provider, "_chat", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate("read test.py", namespace_desc="  read(path) -> str: Read file")
            assert result is not None
            assert "read(" in result

    @pytest.mark.asyncio
    async def test_generate_with_error_feedback(self, provider):
        mock_response = {"message": {"content": "x = read('test.py')\nlen(x)"}}
        with patch.object(provider, "available", return_value=True), \
             patch.object(provider, "_chat", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate("read test.py", namespace_desc="  read(path) -> str: Read file",
                                             error_feedback=["Unknown function: 'open'"])
            assert result is not None
