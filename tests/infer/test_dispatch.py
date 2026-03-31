"""Tests for the inference dispatch pipeline."""

import pytest

from lackpy.infer.dispatch import InferenceDispatcher, GenerationResult


class FakeProvider:
    def __init__(self, name: str, result: str | None, available: bool = True):
        self._name = name
        self._result = result
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return self._available

    async def generate(self, intent, namespace_desc, config=None, error_feedback=None):
        return self._result


class TestDispatchOrder:
    @pytest.mark.asyncio
    async def test_returns_first_valid_result(self):
        d = InferenceDispatcher(providers=[FakeProvider("templates", "x = 1"), FakeProvider("rules", "y = 2")])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.program == "x = 1"
        assert result.provider_name == "templates"

    @pytest.mark.asyncio
    async def test_skips_none_results(self):
        d = InferenceDispatcher(providers=[FakeProvider("templates", None), FakeProvider("rules", "y = 2")])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.program == "y = 2"

    @pytest.mark.asyncio
    async def test_skips_unavailable_providers(self):
        d = InferenceDispatcher(providers=[FakeProvider("templates", "x = 1", available=False), FakeProvider("rules", "y = 2")])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.provider_name == "rules"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        d = InferenceDispatcher(providers=[FakeProvider("templates", None), FakeProvider("rules", None)])
        with pytest.raises(RuntimeError, match="All .* providers failed"):
            await d.generate("test", namespace_desc="", allowed_names=set())


class TestValidation:
    @pytest.mark.asyncio
    async def test_skips_invalid_programs(self):
        d = InferenceDispatcher(providers=[FakeProvider("bad", "import os"), FakeProvider("good", "x = 1")])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.program == "x = 1"
        assert result.provider_name == "good"
