"""Tests for all inference pipeline steps."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.infer.steps.generate import GenerateStep
from lackpy.infer.steps.validate import ValidateStep
from lackpy.infer.steps.cleanup import CleanupStep
from lackpy.infer.steps.few_shot import FewShotCorrectStep
from lackpy.infer.steps.fresh_fix import FreshFixStep
from lackpy.infer.steps.solve import SolveStep
from lackpy.infer.steps.pick import PickStep
from lackpy.infer.steps.restrict import RestrictStep
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


# ── Helpers ──────────────────────────────────────────────────────────

def _make_kit(tool_names=None):
    tool_names = tool_names or ["read", "glob"]
    specs = {
        "read": ToolSpec(name="read", provider="builtin", description="Read file contents",
                         args=[], returns="str", grade_w=1, effects_ceiling=1),
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files matching a glob pattern",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
        "write": ToolSpec(name="write", provider="builtin", description="Write content to a file",
                          args=[], returns="bool", grade_w=3, effects_ceiling=3),
    }
    tools = {n: specs[n] for n in tool_names if n in specs}
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=max((s.grade_w for s in tools.values()), default=0),
                    d=max((s.effects_ceiling for s in tools.values()), default=0)),
        description="\n".join(f"{s.name}() -> {s.returns}: {s.description}" for s in tools.values()),
    )


def _make_trace(step_name="test"):
    return StepTrace(
        step_name=step_name, provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _make_provider(output="glob('**/*.py')"):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


def _ctx_with_program(program, kit=None, valid=None, errors=None):
    kit = kit or _make_kit()
    ctx = StepContext(intent="test", kit=kit)
    ctx.programs.append(ProgramState(
        program=program, intent="test", kit=kit,
        valid=valid, errors=errors or [], trace=_make_trace(),
    ))
    return ctx


# ── GenerateStep ─────────────────────────────────────────────────────

class TestGenerateStep:
    @pytest.mark.asyncio
    async def test_pushes_program_state(self):
        provider = _make_provider("files = glob('**/*.py')")
        step = GenerateStep(provider)
        ctx = StepContext(intent="find python files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current is not None
        assert ctx.current.program == "files = glob('**/*.py')"
        assert ctx.current.valid is None
        assert ctx.current.trace.step_name == "generate"
        assert ctx.current.trace.provider_name == "test"

    @pytest.mark.asyncio
    async def test_sanitizes_and_cleans_output(self):
        raw = "```python\nimport os\nfiles = glob('**/*.py')\n```"
        provider = _make_provider(raw)
        step = GenerateStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert "```" not in ctx.current.program
        assert "import os" not in ctx.current.program

    @pytest.mark.asyncio
    async def test_provider_returns_none(self):
        provider = _make_provider(None)
        step = GenerateStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current is not None
        assert ctx.current.program == ""


# ── ValidateStep ─────────────────────────────────────────────────────

class TestValidateStep:
    @pytest.mark.asyncio
    async def test_valid_program(self):
        ctx = _ctx_with_program("files = glob('**/*.py')\nlen(files)")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is True
        assert ctx.current.errors == []

    @pytest.mark.asyncio
    async def test_invalid_program_with_import(self):
        ctx = _ctx_with_program("import os\nglob('*.py')")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is False
        assert len(ctx.current.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_unknown_function(self):
        ctx = _ctx_with_program("open('file.txt')")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is False

    @pytest.mark.asyncio
    async def test_no_current_is_noop(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current is None


# ── CleanupStep ──────────────────────────────────────────────────────

class TestCleanupStep:
    @pytest.mark.asyncio
    async def test_strips_imports(self):
        ctx = _ctx_with_program("import os\ncontent = read('f.txt')", valid=False, errors=["has imports"])
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert ctx.current.program == "content = read('f.txt')"
        assert ctx.current.trace.step_name == "cleanup"

    @pytest.mark.asyncio
    async def test_pushes_new_program_state(self):
        ctx = _ctx_with_program("import os\nread('f.txt')")
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert len(ctx.programs) == 2

    @pytest.mark.asyncio
    async def test_no_current_is_noop(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert ctx.current is None


# ── FewShotCorrectStep ──────────────────────────────────────────────

class TestFewShotCorrectStep:
    @pytest.mark.asyncio
    async def test_sends_errors_as_feedback(self):
        provider = _make_provider("glob('**/*.py')")
        step = FewShotCorrectStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="import glob", intent="find files", kit=kit,
            valid=False, errors=["Forbidden AST node: Import"], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        provider.generate.assert_called_once()
        call_args = provider.generate.call_args
        assert call_args[1].get("error_feedback") is not None

    @pytest.mark.asyncio
    async def test_pushes_new_program_state(self):
        provider = _make_provider("glob('*.py')")
        step = FewShotCorrectStep(provider)
        kit = _make_kit()
        ctx = _ctx_with_program("bad", kit=kit, valid=False, errors=["err"])
        ctx = await step.run(ctx)
        assert len(ctx.programs) == 2
        assert ctx.current.trace.step_name == "few_shot_correct"


# ── FreshFixStep ─────────────────────────────────────────────────────

class TestFreshFixStep:
    @pytest.mark.asyncio
    async def test_uses_fixer_prompt(self):
        provider = MagicMock()
        provider.name = "test"
        provider._chat = AsyncMock(return_value={"message": {"content": "glob('*.py')"}})
        step = FreshFixStep(provider)
        kit = _make_kit()
        ctx = _ctx_with_program("import glob", kit=kit, valid=False, errors=["Forbidden: Import"])
        ctx = await step.run(ctx)
        assert ctx.current.trace.step_name == "fresh_fix"
        assert len(ctx.programs) == 2

    @pytest.mark.asyncio
    async def test_no_chat_method_tries_create_message(self):
        provider = MagicMock(spec=[])
        provider.name = "test"
        provider._create_message = AsyncMock(return_value="glob('*.py')")
        step = FreshFixStep(provider)
        kit = _make_kit()
        ctx = _ctx_with_program("bad", kit=kit, valid=False, errors=["err"])
        ctx = await step.run(ctx)
        assert ctx.current.program == "glob('*.py')"


# ── SolveStep ────────────────────────────────────────────────────────

class TestSolveStep:
    @pytest.mark.asyncio
    async def test_generates_unconstrained(self):
        code = "import glob\nfiles = glob.glob('**/*.py', recursive=True)\nlen(files)"
        provider = _make_provider(code)
        step = SolveStep(provider)
        ctx = StepContext(intent="find all python files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current is not None
        assert ctx.current.trace.step_name == "solve"
        # Solver preserves standard Python (no cleanup)
        assert "import" in ctx.current.program or "glob" in ctx.current.program

    @pytest.mark.asyncio
    async def test_solve_prompt_mentions_python(self):
        provider = _make_provider("glob.glob('*.py')")
        step = SolveStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert "python" in ctx.current.trace.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_provider_returns_none(self):
        provider = _make_provider(None)
        step = SolveStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current.program == ""


# ── PickStep ─────────────────────────────────────────────────────────

class TestPickStep:
    @pytest.mark.asyncio
    async def test_picks_glob_and_read(self):
        ctx = _ctx_with_program(
            "import glob\nfiles = glob.glob('**/*.py')\n"
            "for f in files:\n    content = read(f)\n    print(len(content))",
            kit=_make_kit(["read", "glob", "write"]),
        )
        step = PickStep()
        ctx = await step.run(ctx)
        derived_tools = set(ctx.current.kit.tools.keys())
        assert "glob" in derived_tools
        assert "read" in derived_tools
        assert "write" not in derived_tools

    @pytest.mark.asyncio
    async def test_picks_from_open_pattern(self):
        ctx = _ctx_with_program(
            "with open('README.md') as f:\n    content = f.read()",
            kit=_make_kit(["read", "glob"]),
        )
        step = PickStep()
        ctx = await step.run(ctx)
        assert "read" in ctx.current.kit.tools

    @pytest.mark.asyncio
    async def test_preserves_program_text(self):
        program = "files = glob('*.py')\nlen(files)"
        ctx = _ctx_with_program(program)
        step = PickStep()
        ctx = await step.run(ctx)
        assert ctx.current.program == program

    @pytest.mark.asyncio
    async def test_no_current_is_noop(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = PickStep()
        ctx = await step.run(ctx)
        assert ctx.current is None


# ── RestrictStep ─────────────────────────────────────────────────────

class TestRestrictStep:
    @pytest.mark.asyncio
    async def test_rewrites_standard_python(self):
        provider = _make_provider("files = glob('**/*.py')\nlen(files)")
        step = RestrictStep(provider)
        kit = _make_kit()
        ctx = _ctx_with_program(
            "import glob\nfiles = glob.glob('**/*.py', recursive=True)\nlen(files)",
            kit=kit,
        )
        ctx = await step.run(ctx)
        assert ctx.current.trace.step_name == "restrict"
        assert ctx.current.program == "files = glob('**/*.py')\nlen(files)"

    @pytest.mark.asyncio
    async def test_restrict_prompt_contains_tools(self):
        provider = _make_provider("glob('*.py')")
        step = RestrictStep(provider)
        kit = _make_kit()
        ctx = _ctx_with_program("import glob\nglob.glob('*.py')", kit=kit)
        ctx = await step.run(ctx)
        prompt = ctx.current.trace.system_prompt
        assert "glob" in prompt
        assert "read" in prompt

    @pytest.mark.asyncio
    async def test_restrict_prompt_contains_original(self):
        original = "import glob\nfiles = glob.glob('*.py')"
        provider = _make_provider("files = glob('*.py')")
        step = RestrictStep(provider)
        ctx = _ctx_with_program(original)
        ctx = await step.run(ctx)
        user_prompt = ctx.current.trace.user_prompt
        assert "glob.glob" in user_prompt
