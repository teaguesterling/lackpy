# Sandbox Service Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire SandboxedInterpreter into LackpyService so policy-driven sandbox constraints are consumed at execution time, remove dead JailCallStrategy code, and add bridge authkey test coverage.

**Architecture:** Per-request sandbox activation in `delegate()` and `run_program()`. When policy constraints are non-empty or the caller passes `sandbox=True`, execution routes through `SandboxedInterpreter` instead of `self._runner.run()`. Tool provisioning classifies kit tools as embedded/bridged/unavailable and wires them through the subprocess strategy.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, multiprocessing.managers (bridge IPC), nsjail-python (optional, runtime only)

**Important:** All pytest commands must include `--rootdir=.` because this is a git worktree and pytest resolves `rootdir` to the parent repo otherwise, breaking the `pythonpath = ["src"]` configuration.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/lackpy/sandbox/strategies.py` | SubprocessStrategy only (JailCallStrategy removed) |
| `src/lackpy/sandbox/__init__.py` | Public exports (JailCallStrategy removed) |
| `src/lackpy/interpreters/sandbox.py` | SandboxedInterpreter: simplified constructor (no strategy param), execute() with tool provisioning |
| `src/lackpy/service.py` | LackpyService: sandbox backend init, sandbox routing in run_program()/delegate() |
| `src/lackpy/kit/toolbox.py` | Toolbox: add `get_provider()` public method |
| `tests/sandbox/test_strategies.py` | Updated: JailCallStrategy tests removed |
| `tests/sandbox/test_sandbox_interpreter.py` | Updated: strategy param removed, provisioning tests added |
| `tests/sandbox/test_bridge.py` | Extended: authkey tests |
| `tests/sandbox/test_exports.py` | Updated: JailCallStrategy export removed |
| `tests/sandbox/test_service_sandbox.py` | New: service-level sandbox routing tests |

---

### Task 1: Remove JailCallStrategy

**Files:**
- Modify: `src/lackpy/sandbox/strategies.py:91-121`
- Modify: `src/lackpy/sandbox/__init__.py`
- Modify: `tests/sandbox/test_strategies.py:102-117`
- Modify: `tests/sandbox/test_exports.py`

- [ ] **Step 1: Delete JailCallStrategy tests**

Remove the `TestJailCallStrategy` class from `tests/sandbox/test_strategies.py`. Delete lines 102-117:

```python
# DELETE this entire class:
class TestJailCallStrategy:
    @pytest.mark.asyncio
    async def test_raises_on_unserializable(self):
        ...
```

The file should end after `TestSubprocessStrategy`.

- [ ] **Step 2: Run tests to verify removal is clean**

Run: `python -m pytest tests/sandbox/test_strategies.py -v --rootdir=.`
Expected: 3 passed (only `TestSubprocessStrategy` tests remain)

- [ ] **Step 3: Delete JailCallStrategy class from strategies.py**

Remove lines 91-121 from `src/lackpy/sandbox/strategies.py` — the entire `JailCallStrategy` class.

Also update the module docstring on line 1:
```python
"""Execution strategies: subprocess worker harness."""
```

The file should contain only the imports and `SubprocessStrategy`.

- [ ] **Step 4: Remove JailCallStrategy from __init__.py exports**

The `__init__.py` does not currently export `JailCallStrategy` (confirmed by reading). No change needed here — but verify the import still works:

Run: `python -c "from lackpy.sandbox.strategies import SubprocessStrategy; print('ok')"`

(Run from project root with `PYTHONPATH=src`)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass (same count as before minus the 1 deleted test)

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/sandbox/strategies.py tests/sandbox/test_strategies.py
git commit -m "refactor(sandbox): remove JailCallStrategy dead code"
```

---

### Task 2: Simplify SandboxedInterpreter (remove strategy dispatch)

**Files:**
- Modify: `src/lackpy/interpreters/sandbox.py:26-84`
- Modify: `tests/sandbox/test_sandbox_interpreter.py`

- [ ] **Step 1: Write test asserting no strategy parameter**

Add a test to `tests/sandbox/test_sandbox_interpreter.py` in `TestSandboxedInterpreterValidation`:

```python
def test_no_strategy_parameter(self):
    from lackpy.interpreters.sandbox import SandboxedInterpreter
    inner = MagicMock()
    inner.name = "python"
    inner.description = "Python interpreter"
    backend = MagicMock()
    si = SandboxedInterpreter(interpreter=inner, backend=backend)
    assert not hasattr(si, '_strategy')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sandbox/test_sandbox_interpreter.py::TestSandboxedInterpreterValidation::test_no_strategy_parameter -v --rootdir=.`
Expected: FAIL — `_strategy` attribute exists

- [ ] **Step 3: Remove strategy parameter and dispatch logic**

Modify `src/lackpy/interpreters/sandbox.py`:

**Constructor** — remove `strategy` param and `self._strategy`:

```python
class SandboxedInterpreter:
    """Decorator that wraps any Interpreter with sandbox execution."""

    def __init__(
        self,
        interpreter: Any,
        backend: SandboxBackend,
        base_config: SandboxBaseConfig | None = None,
    ) -> None:
        self.interpreter = interpreter
        self._backend = backend
        self._base_config = base_config or SandboxBaseConfig()
```

**execute()** — remove strategy dispatch, inline subprocess:

```python
    async def execute(
        self,
        program: str,
        context: ExecutionContext,
        sandbox_constraints: tuple[SandboxConstraint, ...] = (),
        backend_configs: Mapping[str, Any] | None = None,
    ) -> InterpreterExecutionResult:
        backend_configs = backend_configs or EMPTY_MAP

        config = self._resolve_config(sandbox_constraints, backend_configs, context)

        strategy = SubprocessStrategy(backend=self._backend)
        return await strategy.run(
            program=program,
            embedded_sources={},
            bridge_socket=None,
            bridge_authkey=None,
            base_dir=context.base_dir,
            config=config,
        )
```

Also remove the now-unused lazy import of `JailCallStrategy` (the `from ..sandbox.strategies import JailCallStrategy` line inside execute is gone).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/sandbox/test_sandbox_interpreter.py -v --rootdir=.`
Expected: All tests pass

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/interpreters/sandbox.py tests/sandbox/test_sandbox_interpreter.py
git commit -m "refactor(sandbox): simplify SandboxedInterpreter, remove strategy dispatch"
```

---

### Task 3: Add Toolbox.get_provider() public method

**Files:**
- Modify: `src/lackpy/kit/toolbox.py:67-128`
- Test: `tests/test_toolbox.py` (or wherever toolbox tests live)

- [ ] **Step 1: Find existing toolbox tests**

Run: `find tests/ -name '*toolbox*' -o -name '*kit*' | head -10`

Check what test file covers `Toolbox`. If none exists, we'll add the test inline to a sandbox test file.

- [ ] **Step 2: Write the failing test**

Add to the test file that covers Toolbox (or create `tests/test_toolbox.py`):

```python
def test_get_provider_returns_registered_provider():
    from lackpy.kit.toolbox import Toolbox
    from unittest.mock import MagicMock

    toolbox = Toolbox()
    provider = MagicMock()
    provider.name = "builtin"
    toolbox.register_provider(provider)

    result = toolbox.get_provider("builtin")
    assert result is provider


def test_get_provider_returns_none_for_unknown():
    from lackpy.kit.toolbox import Toolbox

    toolbox = Toolbox()
    result = toolbox.get_provider("nonexistent")
    assert result is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_toolbox.py::test_get_provider_returns_registered_provider -v --rootdir=.`
Expected: FAIL — `get_provider` not defined

- [ ] **Step 4: Implement get_provider()**

Add to `src/lackpy/kit/toolbox.py` in class `Toolbox`, after `resolve()`:

```python
    def get_provider(self, name: str) -> Any | None:
        """Return the provider registered under the given name, or None."""
        return self._providers.get(name)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_toolbox.py -v --rootdir=.`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/kit/toolbox.py tests/test_toolbox.py
git commit -m "feat(kit): add Toolbox.get_provider() public method"
```

---

### Task 4: Wire tool provisioning into SandboxedInterpreter.execute()

**Files:**
- Modify: `src/lackpy/interpreters/sandbox.py:53-84`
- Modify: `tests/sandbox/test_sandbox_interpreter.py`

- [ ] **Step 1: Write failing test for tool provisioning (embedded tools)**

Add a new test class to `tests/sandbox/test_sandbox_interpreter.py`:

```python
class TestSandboxedInterpreterProvisioning:
    @pytest.mark.asyncio
    async def test_embedded_tools_passed_to_strategy(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        from lackpy.sandbox.provisioning import ToolProvisionKind

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = None
        backend.compile.return_value = CompilationResult(config=MagicMock(), warnings=[])

        provider = MagicMock()
        provider.name = "builtin"
        provider.get_source.return_value = "def read_file(path): return open(path).read()"

        spec = MagicMock()
        spec.name = "read_file"
        spec.provider = "builtin"

        tool_pairs = {"read_file": (spec, provider)}

        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True))
            mock_strat_cls.return_value = mock_strat

            await si.execute(
                "x = read_file('test.txt')", ctx,
                sandbox_constraints=(),
                tool_pairs=tool_pairs,
            )

        call_kwargs = mock_strat.run.call_args
        assert "read_file" in call_kwargs.kwargs.get("embedded_sources", call_kwargs[1].get("embedded_sources", {}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sandbox/test_sandbox_interpreter.py::TestSandboxedInterpreterProvisioning::test_embedded_tools_passed_to_strategy -v --rootdir=.`
Expected: FAIL — `tool_pairs` not accepted

- [ ] **Step 3: Write failing test for bridged tools**

Add to `TestSandboxedInterpreterProvisioning`:

```python
    @pytest.mark.asyncio
    async def test_bridged_tools_start_bridge(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        from lackpy.sandbox.constraints import BridgedToolPolicy

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = None
        backend.compile.return_value = CompilationResult(config=MagicMock(), warnings=[])

        provider = MagicMock(spec=["name", "resolve"])
        provider.name = "custom"
        provider.resolve.return_value = lambda x: x

        spec = MagicMock()
        spec.name = "custom_tool"
        spec.provider = "custom"

        tool_pairs = {"custom_tool": (spec, provider)}
        bridge_policy = BridgedToolPolicy(allowed=True, allowed_providers=("custom",))

        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls, \
             patch("lackpy.interpreters.sandbox.ToolBridgeManager") as mock_bridge_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True))
            mock_strat_cls.return_value = mock_strat

            mock_bridge = MagicMock()
            mock_bridge.socket_path = Path("/tmp/bridge.sock")
            mock_bridge.authkey = b"\x00" * 32
            mock_bridge.__enter__ = MagicMock(return_value=mock_bridge)
            mock_bridge.__exit__ = MagicMock(return_value=False)
            mock_bridge_cls.return_value = mock_bridge

            await si.execute(
                "custom_tool('test')", ctx,
                sandbox_constraints=(bridge_policy,),
                tool_pairs=tool_pairs,
            )

        mock_bridge_cls.assert_called_once()
        call_kwargs = mock_strat.run.call_args
        bridge_socket = call_kwargs.kwargs.get("bridge_socket", call_kwargs[1].get("bridge_socket"))
        assert bridge_socket is not None
```

- [ ] **Step 4: Write failing test for no tool_pairs backward compat**

Add to `TestSandboxedInterpreterProvisioning`:

```python
    @pytest.mark.asyncio
    async def test_no_tool_pairs_sends_empty_sources(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = None
        backend.compile.return_value = CompilationResult(config=MagicMock(), warnings=[])

        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True))
            mock_strat_cls.return_value = mock_strat

            await si.execute("x = 1", ctx)

        call_kwargs = mock_strat.run.call_args
        embedded = call_kwargs.kwargs.get("embedded_sources", call_kwargs[1].get("embedded_sources", {}))
        assert embedded == {}
        bridge = call_kwargs.kwargs.get("bridge_socket", call_kwargs[1].get("bridge_socket"))
        assert bridge is None
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/sandbox/test_sandbox_interpreter.py::TestSandboxedInterpreterProvisioning -v --rootdir=.`
Expected: FAIL on embedded and bridged tests, the backward-compat test should pass (it matches current behavior)

- [ ] **Step 6: Implement tool provisioning in execute()**

Update `src/lackpy/interpreters/sandbox.py`:

Add imports at the top:
```python
from ..sandbox.constraints import SandboxConstraint, BridgedToolPolicy, merge_constraints
from ..sandbox.provisioning import partition_kit
from ..sandbox.bridge import ToolBridgeManager
```

(The existing import of `SandboxConstraint, merge_constraints` stays; add `BridgedToolPolicy`. Add `partition_kit` and `ToolBridgeManager`.)

Replace the `execute()` method:

```python
    async def execute(
        self,
        program: str,
        context: ExecutionContext,
        sandbox_constraints: tuple[SandboxConstraint, ...] = (),
        backend_configs: Mapping[str, Any] | None = None,
        tool_pairs: dict[str, tuple[Any, Any]] | None = None,
    ) -> InterpreterExecutionResult:
        backend_configs = backend_configs or EMPTY_MAP

        config = self._resolve_config(sandbox_constraints, backend_configs, context)

        embedded_sources: dict[str, str] = {}
        bridge_socket = None
        bridge_authkey = None
        bridge_mgr = None

        if tool_pairs:
            bridge_policy = None
            for c in sandbox_constraints:
                if isinstance(c, BridgedToolPolicy):
                    bridge_policy = c
                    break

            embedded, bridged, unavailable = partition_kit(tool_pairs, bridge_policy)

            for name, (spec, provider) in embedded.items():
                source = provider.get_source(spec)
                if source is not None:
                    embedded_sources[name] = source

            for name in unavailable:
                logger.warning("Tool '%s' unavailable in sandbox", name)

            if bridged:
                bridged_callables = {}
                for name, (spec, provider) in bridged.items():
                    bridged_callables[name] = provider.resolve(spec)
                bridge_mgr = ToolBridgeManager(bridged_callables)
                bridge_mgr.start()
                bridge_socket = bridge_mgr.socket_path
                bridge_authkey = bridge_mgr.authkey

        try:
            strategy = SubprocessStrategy(backend=self._backend)
            return await strategy.run(
                program=program,
                embedded_sources=embedded_sources,
                bridge_socket=bridge_socket,
                bridge_authkey=bridge_authkey,
                base_dir=context.base_dir,
                config=config,
            )
        finally:
            if bridge_mgr is not None:
                bridge_mgr.stop()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/sandbox/test_sandbox_interpreter.py -v --rootdir=.`
Expected: All tests pass

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add src/lackpy/interpreters/sandbox.py tests/sandbox/test_sandbox_interpreter.py
git commit -m "feat(sandbox): wire tool provisioning into SandboxedInterpreter.execute()"
```

---

### Task 5: Wire SandboxedInterpreter into LackpyService

**Files:**
- Modify: `src/lackpy/service.py:89-107,315-346,419-427`
- Create: `tests/sandbox/test_service_sandbox.py`

- [ ] **Step 1: Write failing test for sandbox backend initialization**

Create `tests/sandbox/test_service_sandbox.py`:

```python
"""Tests for sandbox routing in LackpyService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lackpy.config import LackpyConfig
from lackpy.sandbox.config import SandboxBaseConfig


class TestServiceSandboxInit:
    def test_sandbox_backend_created_when_config_enabled(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=True, backend="nsjail"),
            config_dir=tmp_path / ".lackpy",
        )
        with patch("lackpy.service.NsjailBackend") as mock_backend_cls:
            mock_backend_cls.return_value = MagicMock()
            from lackpy.service import LackpyService
            svc = LackpyService(workspace=tmp_path, config=config)
            assert svc._sandbox_backend is not None
            mock_backend_cls.assert_called_once()

    def test_sandbox_backend_none_when_config_disabled(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=False),
            config_dir=tmp_path / ".lackpy",
        )
        from lackpy.service import LackpyService
        svc = LackpyService(workspace=tmp_path, config=config)
        assert svc._sandbox_backend is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py::TestServiceSandboxInit -v --rootdir=.`
Expected: FAIL — `NsjailBackend` not imported in service, `_sandbox_backend` not set

- [ ] **Step 3: Implement sandbox backend init in LackpyService.__init__()**

Modify `src/lackpy/service.py`:

Add after the existing imports (around line 20):

```python
from .sandbox.backends.nsjail import NsjailBackend
```

In `__init__()`, after `self._init_kibitzer()` (around line 118), add:

```python
        self._sandbox_backend = None
        if self._config.sandbox.enabled:
            self._sandbox_backend = NsjailBackend()
```

- [ ] **Step 4: Run init tests to verify they pass**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py::TestServiceSandboxInit -v --rootdir=.`
Expected: PASS

- [ ] **Step 5: Write failing test for run_program sandbox routing**

Add to `tests/sandbox/test_service_sandbox.py`:

```python
class TestServiceRunProgramSandbox:
    @pytest.mark.asyncio
    async def test_sandbox_true_routes_through_sandboxed_interpreter(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=True),
            config_dir=tmp_path / ".lackpy",
        )
        with patch("lackpy.service.NsjailBackend") as mock_backend_cls, \
             patch("lackpy.service.SandboxedInterpreter") as mock_si_cls:
            mock_backend_cls.return_value = MagicMock()
            mock_si = MagicMock()
            mock_si.execute = AsyncMock(return_value=MagicMock(
                success=True, output="hello", error=None,
                output_format="text",
                trace=MagicMock(entries=[], files_read=[], files_modified=[]),
            ))
            mock_si_cls.return_value = mock_si

            from lackpy.service import LackpyService
            svc = LackpyService(workspace=tmp_path, config=config)

            result = await svc.run_program("x = 1", sandbox=True)

            mock_si_cls.assert_called_once()
            mock_si.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_false_uses_runner_directly(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=True),
            config_dir=tmp_path / ".lackpy",
        )
        with patch("lackpy.service.NsjailBackend") as mock_backend_cls:
            mock_backend_cls.return_value = MagicMock()
            from lackpy.service import LackpyService
            svc = LackpyService(workspace=tmp_path, config=config)

            with patch.object(svc._runner, "run") as mock_run:
                mock_run.return_value = MagicMock(
                    success=True, output="hello", error=None,
                    trace=MagicMock(entries=[], files_read=[], files_modified=[]),
                )
                result = await svc.run_program("x = 1", sandbox=False)
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_requested_but_no_backend_raises(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=False),
            config_dir=tmp_path / ".lackpy",
        )
        from lackpy.service import LackpyService
        svc = LackpyService(workspace=tmp_path, config=config)

        with pytest.raises(RuntimeError, match="sandbox"):
            await svc.run_program("x = 1", sandbox=True)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py::TestServiceRunProgramSandbox -v --rootdir=.`
Expected: FAIL — `SandboxedInterpreter` not imported in service, routing logic not present

- [ ] **Step 7: Implement sandbox routing in run_program()**

Modify `src/lackpy/service.py`:

Add import near the top (after the NsjailBackend import):

```python
from .interpreters.sandbox import SandboxedInterpreter
```

Change the `sandbox` parameter type in `run_program()` signature from `sandbox: Any = None` to `sandbox: bool = False`.

Replace the execution block in `run_program()` (lines 340-345):

```python
        if sandbox:
            if self._sandbox_backend is None:
                raise RuntimeError(
                    "Sandbox requested but no backend configured. "
                    "Set [sandbox] enabled=true in .lackpy/config.toml"
                )
            tool_pairs = self._build_tool_pairs(resolved)
            si = SandboxedInterpreter(
                interpreter=None,
                backend=self._sandbox_backend,
                base_config=self._config.sandbox,
            )
            from .interpreters.base import ExecutionContext
            ctx = ExecutionContext(base_dir=self._workspace)
            exec_result = await si.execute(
                program, ctx,
                tool_pairs=tool_pairs,
            )
            return ExecutionResult(
                success=exec_result.success,
                output=exec_result.output,
                error=exec_result.error,
            )

        prev_cwd = os.getcwd()
        try:
            os.chdir(self._workspace)
            return self._runner.run(program, resolved.callables, params=param_values)
        finally:
            os.chdir(prev_cwd)
```

Note: `run_program()` only activates sandbox via the explicit `sandbox=True` flag. Policy-driven activation (via `policy.sandbox_constraints`) only applies in `delegate()`, where policy is already resolved.

Add helper method to `LackpyService`:

```python
    def _build_tool_pairs(self, kit: ResolvedKit) -> dict[str, tuple[Any, Any]]:
        pairs: dict[str, tuple[Any, Any]] = {}
        for name, spec in kit.tools.items():
            provider = self.toolbox.get_provider(spec.provider)
            if provider is not None:
                pairs[name] = (spec, provider)
        return pairs
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py -v --rootdir=.`
Expected: All tests pass

- [ ] **Step 9: Write failing test for delegate sandbox routing**

Add to `tests/sandbox/test_service_sandbox.py`:

```python
class TestServiceDelegateSandbox:
    @pytest.mark.asyncio
    async def test_delegate_with_sandbox_routes_through_si(self, tmp_path):
        config = LackpyConfig(
            sandbox=SandboxBaseConfig(enabled=True),
            config_dir=tmp_path / ".lackpy",
        )
        with patch("lackpy.service.NsjailBackend") as mock_backend_cls, \
             patch("lackpy.service.SandboxedInterpreter") as mock_si_cls:
            mock_backend_cls.return_value = MagicMock()
            mock_si = MagicMock()
            mock_exec_result = MagicMock(
                success=True, output="hello", error=None,
                output_format="text",
            )
            mock_si.execute = AsyncMock(return_value=mock_exec_result)
            mock_si_cls.return_value = mock_si

            from lackpy.service import LackpyService
            svc = LackpyService(workspace=tmp_path, config=config)

            result = await svc.delegate(
                "read a file",
                _program_override="x = read_file('test.txt')",
                sandbox=True,
            )

            mock_si_cls.assert_called_once()
            assert result["success"] is True
```

- [ ] **Step 10: Run test to verify it fails**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py::TestServiceDelegateSandbox -v --rootdir=.`
Expected: FAIL — delegate() doesn't route through sandbox yet

- [ ] **Step 11: Implement sandbox routing in delegate()**

Change the `sandbox` parameter type in `delegate()` signature from `sandbox: Any = None` to `sandbox: bool = False`.

In `delegate()`, replace the execution block (around lines 419-427):

```python
        if sandbox or policy.sandbox_constraints:
            if self._sandbox_backend is None:
                raise RuntimeError(
                    "Sandbox requested but no backend configured. "
                    "Set [sandbox] enabled=true in .lackpy/config.toml"
                )
            tool_pairs = self._build_tool_pairs(resolved)
            si = SandboxedInterpreter(
                interpreter=None,
                backend=self._sandbox_backend,
                base_config=self._config.sandbox,
            )
            from .interpreters.base import ExecutionContext
            ctx = ExecutionContext(base_dir=self._workspace)
            exec_result = await si.execute(
                gen_result.program, ctx,
                sandbox_constraints=policy.sandbox_constraints,
                backend_configs=policy.sandbox_backend_configs,
                tool_pairs=tool_pairs,
            )
        else:
            prev_cwd = os.getcwd()
            try:
                os.chdir(self._workspace)
                exec_result_raw = self._runner.run(
                    gen_result.program, resolved.callables, params=param_values,
                    kibitzer_session=self._kibitzer,
                )
            finally:
                os.chdir(prev_cwd)
            exec_result = exec_result_raw
```

Note: The rest of `delegate()` uses `exec_result.success`, `exec_result.output`, `exec_result.error`, and `exec_result.trace`. Both `ExecutionResult` (from runner) and `InterpreterExecutionResult` (from sandbox) have `success`, `output`, `error`. But `ExecutionResult` has `.trace` while `InterpreterExecutionResult` has `.metadata`. For the sandbox path, we need to build a compatible trace:

After the sandbox `execute()` call, convert the result:

```python
            # Convert InterpreterExecutionResult to ExecutionResult for downstream
            from .run.trace import Trace
            exec_result = ExecutionResult(
                success=exec_result.success,
                output=exec_result.output,
                error=exec_result.error,
                trace=Trace(),
            )
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `python -m pytest tests/sandbox/test_service_sandbox.py -v --rootdir=.`
Expected: All tests pass

- [ ] **Step 13: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass

- [ ] **Step 14: Commit**

```bash
git add src/lackpy/service.py tests/sandbox/test_service_sandbox.py
git commit -m "feat(sandbox): wire SandboxedInterpreter into LackpyService delegate/run_program"
```

---

### Task 6: Bridge authkey tests

**Files:**
- Modify: `tests/sandbox/test_bridge.py`
- Modify: `tests/sandbox/test_strategies.py`

- [ ] **Step 1: Write matching-authkey test**

Add to `tests/sandbox/test_bridge.py`, in class `TestToolBridgeManager`:

```python
    def test_matching_authkey_allows_connection(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager, bridge_client
        def add(a: int, b: int) -> int:
            return a + b
        with ToolBridgeManager(callables={"add": add}, socket_dir=tmp_path) as mgr:
            client = bridge_client(mgr.socket_path, authkey=mgr.authkey)
            assert client.call("add", 2, 3) == 5
            assert "add" in client.list_tools()
```

- [ ] **Step 2: Run test to verify it passes (this is coverage, not TDD)**

Run: `python -m pytest tests/sandbox/test_bridge.py::TestToolBridgeManager::test_matching_authkey_allows_connection -v --rootdir=.`
Expected: PASS (the bridge already works with matching authkeys)

- [ ] **Step 3: Write wrong-authkey test**

Add to `tests/sandbox/test_bridge.py`, in class `TestToolBridgeManager`:

```python
    def test_wrong_authkey_rejected(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager, bridge_client
        with ToolBridgeManager(callables={"noop": lambda: None}, socket_dir=tmp_path) as mgr:
            wrong_key = b"\xff" * 32
            assert wrong_key != mgr.authkey
            with pytest.raises((ConnectionRefusedError, EOFError, OSError, Exception)):
                bridge_client(mgr.socket_path, authkey=wrong_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/sandbox/test_bridge.py::TestToolBridgeManager::test_wrong_authkey_rejected -v --rootdir=.`
Expected: PASS — multiprocessing.managers rejects wrong authkey with `AuthenticationError`

- [ ] **Step 5: Write authkey-in-request test**

Add to `tests/sandbox/test_strategies.py`, in class `TestSubprocessStrategy`:

```python
    @pytest.mark.asyncio
    async def test_bridge_authkey_included_in_request(self):
        from lackpy.sandbox.strategies import SubprocessStrategy
        from lackpy.sandbox.backend import SandboxResult
        from lackpy.sandbox._worker import read_request

        mock_backend = MagicMock()
        mock_result = SandboxResult(
            returncode=0, stdout=b"", stderr=b"",
            timed_out=False, oom_killed=False,
        )
        mock_backend.run = AsyncMock(return_value=mock_result)

        strategy = SubprocessStrategy(backend=mock_backend)
        authkey = b"\xab\xcd" * 16  # 32 bytes

        with tempfile.TemporaryDirectory() as td:
            io_dir = Path(td)
            result_data = {
                "success": True, "output": None,
                "output_format": "none", "error": None,
                "duration_ms": 0.0, "metadata": {},
            }
            (io_dir / "result.json").write_text(json.dumps(result_data))

            await strategy.run(
                program="x = 1",
                embedded_sources={"read_file": "def read_file(p): pass"},
                bridge_socket=Path("/tmp/bridge.sock"),
                bridge_authkey=authkey,
                base_dir=Path("/workspace"),
                config=MagicMock(),
                _io_dir=io_dir,
            )

        request = read_request(io_dir)
        assert request["bridge_authkey"] == authkey.hex()
        assert request["bridge_socket"] == "/tmp/bridge.sock"
        assert "read_file" in request["embedded_sources"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/sandbox/test_strategies.py::TestSubprocessStrategy::test_bridge_authkey_included_in_request -v --rootdir=.`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add tests/sandbox/test_bridge.py tests/sandbox/test_strategies.py
git commit -m "test(sandbox): add bridge authkey tests (matching, wrong key, request payload)"
```

---

### Task 7: Final cleanup and verification

**Files:**
- Modify: `tests/sandbox/test_exports.py` (if needed)
- Review: all modified files

- [ ] **Step 1: Verify exports test still passes**

Run: `python -m pytest tests/sandbox/test_exports.py -v --rootdir=.`
Expected: PASS — no exports were added/removed that affect this test

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q --rootdir=. --ignore=tests/literate --ignore=tests/eval`
Expected: All tests pass, count should be original 401 minus 1 (deleted JailCallStrategy test) plus new tests (roughly 10-12 new)

- [ ] **Step 3: Verify no regressions in service tests**

Run: `python -m pytest tests/test_service.py tests/test_config.py tests/test_cli.py -v --rootdir=.`
Expected: All pass

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "chore(sandbox): final integration cleanup"
```
