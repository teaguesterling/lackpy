"""Virtual / harness-provided tools (RFC 0002 §7, increment 4)."""

from lackpy.config import LackpyConfig
from lackpy.service import LackpyService

VT = [{
    "name": "ping",
    "description": "harness ping",
    "returns": "str",
    "grade_w": 2,
    "effects_ceiling": 2,
}]


def _svc(tmp_path, resolver):
    return LackpyService(
        workspace=tmp_path,
        config=LackpyConfig(virtual_tools=VT),
        harness_resolver=resolver,
    )


def test_discover_sets_virtual_provider_and_grade():
    from lackpy.sources.virtual import VirtualToolSource

    spec = VirtualToolSource(VT, lambda n: None).discover()[0]
    assert spec.provider == "virtual"
    assert (spec.grade_w, spec.effects_ceiling) == (2, 2)


async def test_virtual_tool_runs_when_harness_offers_it(tmp_path):
    svc = _svc(tmp_path, lambda n: (lambda: "pong") if n == "ping" else None)
    assert "ping" in {t["name"] for t in svc.toolbox_list()}  # declared/registered
    res = await svc.run_program("p = ping()\np", kit=["ping"])
    assert res.success, res.error
    assert res.output == "pong"


async def test_gate_hides_unavailable_virtual_tool(tmp_path):
    # Resolver never offers it -> gated out of the kit -> program fails validation.
    svc = _svc(tmp_path, lambda n: None)
    res = await svc.run_program("p = ping()\np", kit=["ping"])
    assert not res.success
    assert "Validation failed" in (res.error or "")


async def test_no_resolver_gates_all_virtual_tools(tmp_path):
    svc = _svc(tmp_path, None)
    res = await svc.run_program("p = ping()\np", kit=["ping"])
    assert not res.success
    assert "Validation failed" in (res.error or "")


async def test_call_time_withdrawal_raises_into_failed_result(tmp_path):
    # Available at the gate (call 1), withdrawn by the time the proxy calls (call 2).
    calls = {"n": 0}

    def resolver(name):
        if name != "ping":
            return None
        calls["n"] += 1
        return (lambda: "pong") if calls["n"] == 1 else None

    svc = _svc(tmp_path, resolver)
    res = await svc.run_program("p = ping()\np", kit=["ping"])
    assert not res.success
    assert "unavailable" in (res.error or "")
