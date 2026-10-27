"""The Plan 2 gate: one test per item of design §8, "Carried into Plan 2" (items 1-9).

Items 1-9 are named ``test_gate_N_*``. ``test_review_focus_5_*`` follows them.
Programs run through ``conftest.run_in_lackpy`` (validate, then the runner with a
PathPolicy) unless a test must bypass the validator to prove the runtime closes a
route the static check also closes; those use ``RestrictedRunner`` directly and
say so. Since Task 7 the helper runs ``LackpyService.run_program``.
"""

import asyncio
import collections
import datetime
import sys
import threading
from pathlib import Path

import pytest

from lackpy.lang import validate
from lackpy.run.bridge import AsyncBridge
from lackpy.run.refusal import LackpyRefusal
from lackpy.run.paths import load_policy
from lackpy.run.runner import RestrictedRunner

from .conftest import run_in_lackpy

HERE = Path(__file__).parent


def _raw(program, **kw):
    """Run with the validator bypassed (runtime defence on its own)."""
    return RestrictedRunner().run(program, **kw)


# --- 1. module proxies close statistics.sys and Path('x').parser.os ----------

def test_gate_1_statistics_sys_closed_statically_and_at_runtime():
    program = "import statistics\nstatistics.sys"
    assert not validate(program, allowed_names=set()).valid
    result = _raw(program)  # validator bypassed: the proxy alone must refuse
    assert not result.success
    assert result.output is None


@pytest.mark.skipif(sys.version_info < (3, 13), reason="Path.parser exists only on Python 3.13+; "
                    "before that the stand-in refuses the name as unknown, which proves nothing")
@pytest.mark.parametrize("program", [
    "from pathlib import Path\nPath('x').parser.os",
    "from pathlib import Path\nPath('x').parser",
])
def test_gate_1_path_parser_closed_at_runtime(program, tmp_path, tmp_home):
    # The validator ADMITS these (measured), so the runtime stand-in is the only guard.
    assert validate(program, allowed_names=set()).valid
    assert hasattr(Path("x"), "parser")
    result = run_in_lackpy(program, tmp_path, tmp_home)
    assert not result.success
    assert result.output is None
    assert "Path.parser" in result.error, result.error


# --- 2. no setattr on member bindings ----------------------------------------

@pytest.mark.parametrize("program", [
    "from collections import Counter\nC = Counter\nC.most_common = len",
    "from pathlib import Path\n[Path][0].read_text = print",
])
def test_gate_2_setattr_on_member_bindings_refused(program, tmp_path, tmp_home):
    """Refused by the validator (attr-store). Not also run with the validator bypassed:
    collections serves the real classes (ruling), so a bypassed run would mutate the
    host's ``collections.Counter``. The Path stand-in does refuse at runtime and is
    run for the second program."""
    assert not validate(program, allowed_names=set()).valid
    if "Path" in program:
        result = _raw(program, path_policy=load_policy(tmp_path, home=tmp_home, managed=None))
        assert not result.success


# --- 3. the trace records a BaseException refusal ----------------------------

def test_gate_3_trace_records_a_refusal(tmp_path, tmp_home):
    (tmp_path / "ws").mkdir()
    ws = tmp_path / "ws"
    result = run_in_lackpy('from pathlib import Path\nPath("../outside.txt").read_text()', ws, tmp_home)
    assert not result.success
    assert result.trace.entries, "the refusal left no trace entry"
    entry = result.trace.entries[-1]
    assert entry.success is False
    assert entry.error


# --- 4. run never raises; validate never raises -------------------------------

@pytest.mark.parametrize("depth", [3_000, 50_000, 500_000])
def test_gate_4_run_never_raises_on_deep_expressions(depth):
    result = _raw("x = " + "-" * depth + "1\n")  # must return a result, pass or fail
    assert result.success or result.error


@pytest.mark.parametrize("program", [
    "x = " + "-" * 50_000 + "1\n",
    "x = " + "-" * 500_000 + "1\n",
    "x = " + "+".join(["1"] * 200_000) + "\n",
    "x = " + "[" * 5_000 + "]" * 5_000 + "\n",
], ids=["unary-50k", "unary-500k", "binop-200k", "lists-5k"])
def test_gate_4_validate_never_raises_on_deep_expressions(program):
    verdict = validate(program, allowed_names=set())  # must return, not raise
    assert verdict.valid or verdict.violations


def test_gate_4_too_deep_is_a_safety_parse_violation():
    verdict = validate("x = " + "-" * 50_000 + "1\n", allowed_names=set())
    assert not verdict.valid
    assert [v.rule_id for v in verdict.violations] == ["safety.parse"]
    assert "nesting depth" in verdict.errors[0]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python 3.11's parser and compiler stop at "
                    "the recursion limit (measured: 992 unary minuses validate and run, 2,991 parse); "
                    "no program deeper than the limit is valid there")
def test_gate_4_deep_valid_program_validates_then_runs(tmp_path, tmp_home):
    # run_in_lackpy raises Invalid if validate() rejects, so reaching the result proves both halves.
    result = run_in_lackpy("x = " + "-" * 3_000 + "1\n", tmp_path, tmp_home)
    assert result.success, result.error
    assert result.variables == {"x": 1}


# --- 5. a hung tool is a refusal the program cannot catch ---------------------
#
# Parked item 6: rewritten without a kit namespace. No program can reach a bridged
# call since the kit went (Task 7), so the gate is the bridge's own contract: a
# timeout is a LackpyRefusal, which no handler the validator admits can catch.

@pytest.mark.parametrize("handler", ["except TimeoutError:", "except Exception:"])
def test_gate_5_timeout_cannot_be_caught(handler):
    bridge = AsyncBridge()
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    bridge.loop = loop
    calls = []

    async def hang():
        await asyncio.sleep(10)

    def hung_call(n):
        calls.append(n)
        return bridge.call_sync(hang(), timeout=0.1)

    code = f"try:\n    y = hung_call(1)\n{handler}\n    y = hung_call(2)\n"
    try:
        with pytest.raises(LackpyRefusal, match="waiting on a tool call past its timeout"):
            exec(code, {"hung_call": hung_call})  # noqa: S102
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
    assert calls == [1], "the handler retried the hung call"


def test_gate_5_no_program_reaches_a_bridged_call():
    # The runner takes no tool namespace, so no bridged callable is bound.
    with pytest.raises(TypeError):
        RestrictedRunner().run("x = 1", {"tool": lambda: 0})


def test_gate_5_bare_except_is_rejected_statically():
    """A bare ``except:`` catches ``BaseException`` in CPython, so it would catch the timeout
    refusal at run time (measured: the handler retries the hung tool when the validator is
    bypassed). Only the static rule guards it; the runner cannot. Reported in task-6-report.md."""
    program = "try:\n    y = tool(1)\nexcept:\n    y = tool(2)\n"
    assert not validate(program, allowed_names={"tool"}).valid


# --- 6. __file__ is defined deliberately --------------------------------------

def test_gate_6_file_is_defined(tmp_path, tmp_home):
    program = "f = __file__\nf"
    assert validate(program, allowed_names=set()).valid
    assert run_in_lackpy(program, tmp_path, tmp_home).output == "<stdin>"
    named = run_in_lackpy(program, tmp_path, tmp_home, program_file="/w/prog.py")
    assert named.output == "/w/prog.py"


def test_gate_6_path_of_file_does_not_read_the_host(tmp_path, tmp_home):
    program = "from pathlib import Path\nPath(__file__).read_text()"
    assert validate(program, allowed_names=set()).valid
    result = run_in_lackpy(program, tmp_path, tmp_home)
    assert not result.success
    assert "No such file" in result.error and "'<stdin>'" in result.error, result.error


# --- 7. variables carry data, never functions ---------------------------------

def test_gate_7_variables_carry_no_function_objects(tmp_path, tmp_home):
    program = ("import json\nfrom pathlib import Path\n"
               "def f():\n    return 1\n"
               "g = len\nj = json.loads\np = Path('x')\nk = Path\n"
               "n = 3\ns = 'a'\nxs = [1, (2, 3)]\n")
    result = run_in_lackpy(program, tmp_path, tmp_home)
    assert result.success, result.error
    assert result.variables == {"n": 3, "s": "a", "xs": [1, (2, 3)]}


# --- 8. datetime.UTC ----------------------------------------------------------

def test_gate_8_datetime_utc_runs(tmp_path, tmp_home):
    """The runtime needs Python >= 3.11, where ``datetime.UTC`` exists.

    The static validator's Python 3.10 claim belongs to ``lackpy-lang`` and is
    out of scope here.
    """
    assert sys.version_info >= (3, 11)
    program = "from datetime import UTC\nUTC.utcoffset(None).total_seconds()"
    result = run_in_lackpy(program, tmp_path, tmp_home)
    assert result.success, result.error
    assert result.output == 0.0
    assert datetime.UTC is not None


# --- 9. the fail-closed import test is replaced -------------------------------

def test_gate_9_fail_closed_import_test_is_replaced(tmp_path, tmp_home):
    assert not (HERE / "test_import_fail_closed.py").exists()
    assert (HERE / "test_import_proxies.py").exists()
    # What it pinned still holds for what is not approved ...
    assert run_in_lackpy("import json\njson.loads('1')", tmp_path, tmp_home).success
    assert not _raw("import os").success
    # ... and pathlib still fails closed with no policy.
    assert not _raw("from pathlib import Path\nPath('x')").success


# --- Review Focus 5: consecutive runs share no module or class state -----------

def test_review_focus_5_consecutive_runs_share_no_state(tmp_path, tmp_home):
    """What is true: collections serves the REAL Counter/defaultdict/deque/OrderedDict
    (ruling), so the guard is static: attribute stores are rejected by the validator
    and ``setattr`` is not in the namespace. Run 1 therefore cannot change a class or
    module that run 2 sees, and run 2 sees none of run 1's globals. This test asserts
    those facts; it does not claim per-run copies of the classes.
    """
    before = {n: dict(vars(getattr(collections, n)))
              for n in ("Counter", "defaultdict", "deque", "OrderedDict")}

    # The routes to mutate shared state are closed.
    for attack in ("from collections import Counter\nCounter.most_common = len",
                   "import json\njson.loads = len",
                   "from collections import Counter\nsetattr(Counter, 'x', 1)",
                   "from collections import Counter\nCounter.__dict__['x'] = 1"):
        assert not validate(attack, allowed_names=set()).valid, attack
    # With the validator bypassed, only the routes that need no attribute store are safe
    # to run: a bypassed ``Counter.most_common = len`` WOULD mutate the real class
    # (measured; that is the ruled static-only guard, reported in task-6-report.md).
    for attack in ("from collections import Counter\nsetattr(Counter, 'x', 1)",
                   "from collections import Counter\nCounter.__dict__['x'] = 1"):
        assert not _raw(attack).success, attack

    run1 = ("import json\nfrom collections import Counter\nleaked = 41\n"
            "c = Counter('aab')\nc.update('b')\n")
    assert run_in_lackpy(run1, tmp_path, tmp_home).success

    run2 = run_in_lackpy("leaked", tmp_path, tmp_home)
    assert not run2.success and "leaked" in run2.error  # no global carried over
    run3 = run_in_lackpy("import json\nfrom collections import Counter\nj = json.dumps([1])\n"
                         "len(Counter('abc'))", tmp_path, tmp_home)
    assert run3.success and run3.output == 3

    after = {n: dict(vars(getattr(collections, n)))
             for n in ("Counter", "defaultdict", "deque", "OrderedDict")}
    assert after == before
