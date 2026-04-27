"""Tests for the LangGraph node factory."""

from __future__ import annotations

import pytest

from langchain_lackpy._node import make_node


class TestMakeNode:
    def test_returns_callable(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="lackpy_result",
        )
        assert callable(node)

    @pytest.mark.asyncio
    async def test_reads_intent_from_state(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="lackpy_result",
        )
        result = await node({"intent": "read config.yaml"})
        mock_service.delegate.assert_called_once()
        call_kwargs = mock_service.delegate.call_args[1]
        assert call_kwargs["intent"] == "read config.yaml"

    @pytest.mark.asyncio
    async def test_passes_kit_config(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file", "write_file"],
            intent_key="intent",
            result_key="result",
        )
        await node({"intent": "do something"})
        call_kwargs = mock_service.delegate.call_args[1]
        assert call_kwargs["kit"] == ["read_file", "write_file"]

    @pytest.mark.asyncio
    async def test_returns_result_under_correct_key(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="task",
            result_key="output",
        )
        result = await node({"task": "read test.txt"})
        assert "output" in result
        assert result["output"]["success"] is True

    @pytest.mark.asyncio
    async def test_custom_keys(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="my_intent",
            result_key="my_result",
        )
        result = await node({"my_intent": "hello"})
        assert "my_result" in result

    @pytest.mark.asyncio
    async def test_missing_intent_key_raises(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="result",
        )
        with pytest.raises(KeyError, match="intent"):
            await node({"wrong_key": "hello"})
