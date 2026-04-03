"""Pick step: analyze solver output AST to derive which tools are needed."""

from __future__ import annotations

import ast
import time

from ..context import ProgramState, StepContext, StepTrace
from ...kit.registry import ResolvedKit
from ...lang.grader import Grade


# Patterns in standard Python that map to lackpy tools
_TOOL_PATTERNS: dict[str, list[str]] = {
    "glob": ["glob.glob", "glob.iglob", "glob("],
    "read": ["open(", ".read(", "readlines(", "read("],
    "write": [".write(", "write("],
    "edit": ["edit(", ".replace("],
}


def _extract_tool_names(program: str, available_tools: dict) -> set[str]:
    """Identify which available tools the program uses or implies."""
    used = set()

    # Check direct calls via AST
    try:
        tree = ast.parse(program)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in available_tools:
                    used.add(node.func.id)
    except SyntaxError:
        pass

    # Check standard Python patterns
    for tool_name, patterns in _TOOL_PATTERNS.items():
        if tool_name in available_tools:
            for pattern in patterns:
                if pattern in program:
                    used.add(tool_name)
                    break

    return used


class PickStep:
    """Analyze the current program to derive which tools it needs.

    Pushes a new ProgramState with the same program text but a
    derived kit containing only the tools the program actually uses.
    Does not require an LLM -- pure AST and pattern analysis.
    """

    name = "pick"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        available = ctx.kit.tools
        used = _extract_tool_names(ctx.current.program, available)

        # Build derived kit with only the tools that were used
        derived_tools = {name: spec for name, spec in available.items() if name in used}
        derived_callables = {name: cb for name, cb in ctx.kit.callables.items() if name in used}

        if derived_tools:
            max_w = max(s.grade_w for s in derived_tools.values())
            max_d = max(s.effects_ceiling for s in derived_tools.values())
        else:
            max_w, max_d = 0, 0

        derived_kit = ResolvedKit(
            tools=derived_tools,
            callables=derived_callables,
            grade=Grade(w=max_w, d=max_d),
            description="\n".join(
                f"{s.name}({', '.join(a.name for a in s.args)}) -> {s.returns}: {s.description}"
                for s in derived_tools.values()
            ),
        )

        elapsed = (time.perf_counter() - start) * 1000

        ctx.programs.append(ProgramState(
            program=ctx.current.program,
            intent=ctx.current.intent,
            kit=derived_kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=None,
                model=None,
                system_prompt=None,
                user_prompt=None,
                raw_output=None,
                duration_ms=elapsed,
            ),
        ))
        return ctx
