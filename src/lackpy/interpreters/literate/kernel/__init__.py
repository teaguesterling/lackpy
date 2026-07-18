"""Incremental cell execution kernel.

Two execution paths exist:

  Batch (LiterateInterpreter.execute in ../__init__.py):
    Uses LightweightKernel directly. No recovery, no plugins,
    no streaming. Suited for tests and simple one-shot execution.

  Streaming (StreamingDriver):
    StreamingCellParser detects fence boundaries in partial model
    output, yielding Cell objects as they complete. The driver
    feeds each cell to LightweightKernel and orchestrates recovery
    (via RecoveryHandler) and plugin advice (via ExecutionPlugin).

Both paths share the same kernel: LightweightKernel compiles each
cell via compiler._COMPILERS, runs static_analysis.check_cell()
(compile check + AST name resolution), then evaluates the result
in a persistent namespace dict.
"""

from .driver import CellExecutionEvent, StreamingDriver
from .interface import CellResult, KernelInterface
from .ledger import AIDR_LEDGER_COLUMNS, Ledger, LedgerEntry
from .lightweight import LightweightKernel
from .plugins import ExecutionPlugin, PluginAdvice, merge_advice
from .recovery import (
    InferenceRecoveryHandler,
    NoRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
    RecoveryHandler,
)
from .streaming_parser import StreamingCellParser

__all__ = [
    "AIDR_LEDGER_COLUMNS",
    "CellExecutionEvent",
    "CellResult",
    "ExecutionPlugin",
    "InferenceRecoveryHandler",
    "KernelInterface",
    "Ledger",
    "LedgerEntry",
    "LightweightKernel",
    "NoRecoveryHandler",
    "PluginAdvice",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryHandler",
    "StreamingCellParser",
    "StreamingDriver",
    "merge_advice",
]
