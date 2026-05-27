"""Tests for the consolidated interpreter base types (RFC 0001 Addendum E prototype):
DelegatingInterpreter (the composite-delegating shape) and IncrementalInterpreter (lifted
from literate's KernelInterface)."""
import asyncio
import dataclasses

import pytest

from lackpy.interpreters.base import (
    DelegatingInterpreter,
    ExecutionContext,
    IncrementalInterpreter,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)


class _RecordingSub:
    """A fake sub-interpreter that records the context it was handed."""
    name = "sub"
    description = "fake"

    def __init__(self):
        self.seen_ctx = None

    def validate(self, program, context):
        self.seen_ctx = context
        return InterpreterValidationResult(valid=True)

    async def execute(self, program, context):
        self.seen_ctx = context
        return InterpreterExecutionResult(success=True, output="ok", output_format="text")


class _Tagging(DelegatingInterpreter):
    name = "tagging"
    description = "wraps a sub, tags the base_dir into config"

    def __init__(self, sub):
        self.sub = sub

    def transform_context(self, context):
        return dataclasses.replace(context, config={**context.config, "wrapped": True})


def test_delegating_transforms_context_and_delegates_validate():
    sub = _RecordingSub()
    interp = _Tagging(sub)
    res = interp.validate("prog", ExecutionContext())
    assert res.valid                                   # delegated to sub
    assert sub.seen_ctx.config["wrapped"] is True      # transform_context applied


def test_delegating_annotates_result_with_its_own_name():
    sub = _RecordingSub()
    interp = _Tagging(sub)
    res = asyncio.run(interp.execute("prog", ExecutionContext()))
    assert res.success and res.output == "ok"          # sub's result passes through
    assert res.metadata["interpreter"] == "tagging"    # annotated as the outer interpreter
    assert sub.seen_ctx.config["wrapped"] is True


def test_plucker_is_a_delegating_interpreter():
    from lackpy.interpreters.python import PythonInterpreter
    from lackpy.interpreters.plucker import PluckerInterpreter

    p = PluckerInterpreter()
    assert isinstance(p, DelegatingInterpreter)
    assert p.name == "plucker"
    assert isinstance(p.sub, PythonInterpreter)        # delegates to restricted-python


def test_literate_kernel_conforms_to_IncrementalInterpreter():
    """The lift: literate's kernel already satisfies the runtime-level incremental
    contract (execute_cell + get_namespace), proving the abstraction is consolidation,
    not invention."""
    from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel

    assert isinstance(LightweightKernel(), IncrementalInterpreter)
