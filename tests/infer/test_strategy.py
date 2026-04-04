"""Tests for InferenceStrategy, OneShotStrategy, and SPMStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.strategy import OneShotStrategy, SPMStrategy, STRATEGIES
from lackpy.infer.context import StepContext
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "find_files": ToolSpec(name="find_files", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1), description="find_files(pattern) -> list[str]: Find files",
    )


def _make_provider(output="find_files('**/*.py')"):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestOneShotStrategy:
    def test_name(self):
        assert OneShotStrategy.name == "1-shot"

    def test_build_returns_step(self):
        provider = _make_provider()
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        assert hasattr(step, "run")

    @pytest.mark.asyncio
    async def test_valid_program_succeeds(self):
        provider = _make_provider("files = find_files('**/*.py')\nlen(files)")
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find python files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current.valid is True
        assert "find_files" in ctx.current.program

    @pytest.mark.asyncio
    async def test_invalid_program_tries_cleanup(self):
        provider = _make_provider("import os\nglob('**/*.py')")
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        # Cleanup should strip the import
        assert ctx.current.valid is True
        assert "import" not in ctx.current.program


class TestSPMStrategy:
    def test_name(self):
        assert SPMStrategy.name == "spm"

    def test_build_returns_step(self):
        provider = _make_provider()
        strategy = SPMStrategy()
        step = strategy.build(provider)
        assert hasattr(step, "run")

    def test_registered(self):
        assert "spm" in STRATEGIES

    @pytest.mark.asyncio
    async def test_spm_end_to_end_with_mock(self):
        call_count = 0

        async def mock_generate(intent, namespace_desc, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "import glob\nfiles = glob.find_files('**/*.py')\nlen(files)"
            else:
                return "files = find_files('**/*.py')\nlen(files)"

        provider = MagicMock()
        provider.name = "test"
        provider.generate = AsyncMock(side_effect=mock_generate)

        strategy = SPMStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find python files and count them", kit=_make_kit())
        ctx = await step.run(ctx)

        assert ctx.current.valid is True
        assert "import" not in ctx.current.program
        assert "find_files(" in ctx.current.program
        assert call_count >= 2


class TestStrategiesRegistry:
    def test_one_shot_registered(self):
        assert "1-shot" in STRATEGIES

    def test_spm_registered(self):
        assert "spm" in STRATEGIES

    def test_registry_values_are_classes(self):
        for name, cls in STRATEGIES.items():
            assert hasattr(cls, "name")
            assert hasattr(cls, "build")
