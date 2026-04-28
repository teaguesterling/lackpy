"""Tests for sandbox constraint types and merge logic."""

from __future__ import annotations

import dataclasses
import pytest


class TestConstraintTypes:
    def test_base_is_frozen(self):
        from lackpy.sandbox.constraints import SandboxConstraint
        c = SandboxConstraint()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.x = 1  # type: ignore[attr-defined]

    def test_memory_limit(self):
        from lackpy.sandbox.constraints import MemoryLimit
        m = MemoryLimit(amount=512, unit="MB")
        assert m.amount == 512
        assert m.unit == "MB"
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.amount = 1024

    def test_time_limit(self):
        from lackpy.sandbox.constraints import TimeLimit
        t = TimeLimit(seconds=120)
        assert t.seconds == 120

    def test_pid_limit(self):
        from lackpy.sandbox.constraints import PidLimit
        p = PidLimit(max_pids=16)
        assert p.max_pids == 16

    def test_cpu_limit(self):
        from lackpy.sandbox.constraints import CpuLimit
        c = CpuLimit(ms_per_sec=500)
        assert c.ms_per_sec == 500

    def test_network_access(self):
        from lackpy.sandbox.constraints import NetworkAccess
        n = NetworkAccess(allow=False)
        assert n.allow is False

    def test_network_restriction(self):
        from lackpy.sandbox.constraints import NetworkRestriction
        n = NetworkRestriction(
            allowed_destinations=("10.0.0.0/8",),
            allowed_protocols=("tcp",),
        )
        assert n.allowed_destinations == ("10.0.0.0/8",)

    def test_filesystem_mount(self):
        from lackpy.sandbox.constraints import FilesystemMount
        f = FilesystemMount(path="/workspace", writable=True)
        assert f.path == "/workspace"
        assert f.writable is True

    def test_readonly_root_default(self):
        from lackpy.sandbox.constraints import ReadonlyRoot
        r = ReadonlyRoot()
        assert r.enabled is True

    def test_seccomp_logging(self):
        from lackpy.sandbox.constraints import SeccompLogging
        s = SeccompLogging(enabled=True)
        assert s.enabled is True

    def test_seccomp_policy(self):
        from lackpy.sandbox.constraints import SeccompPolicyConstraint
        s = SeccompPolicyConstraint(policy_string="ALLOW { read, write }")
        assert "read" in s.policy_string

    def test_user_mapping(self):
        from lackpy.sandbox.constraints import UserMapping
        u = UserMapping(inside_id=0, outside_id=1000)
        assert u.count == 1

    def test_bridged_tool_policy_defaults(self):
        from lackpy.sandbox.constraints import BridgedToolPolicy
        b = BridgedToolPolicy()
        assert b.allowed is False
        assert b.allowed_tools == ()
        assert b.allowed_providers == ()

    def test_all_are_sandbox_constraint_subclasses(self):
        from lackpy.sandbox.constraints import (
            SandboxConstraint, MemoryLimit, TimeLimit, PidLimit, CpuLimit,
            NetworkAccess, NetworkRestriction, FilesystemMount, ReadonlyRoot,
            SeccompLogging, SeccompPolicyConstraint, UserMapping, BridgedToolPolicy,
        )
        for cls in [
            MemoryLimit, TimeLimit, PidLimit, CpuLimit,
            NetworkAccess, NetworkRestriction, FilesystemMount, ReadonlyRoot,
            SeccompLogging, SeccompPolicyConstraint, UserMapping, BridgedToolPolicy,
        ]:
            assert issubclass(cls, SandboxConstraint)


class TestConstraintMerge:
    def test_merge_memory_takes_minimum(self):
        from lackpy.sandbox.constraints import MemoryLimit, merge_constraints
        constraints = (MemoryLimit(amount=512, unit="MB"), MemoryLimit(amount=256, unit="MB"))
        merged = merge_constraints(constraints)
        mems = [c for c in merged if isinstance(c, MemoryLimit)]
        assert len(mems) == 1
        assert mems[0].amount == 256

    def test_merge_memory_converts_units(self):
        from lackpy.sandbox.constraints import MemoryLimit, merge_constraints
        constraints = (MemoryLimit(amount=1, unit="GB"), MemoryLimit(amount=512, unit="MB"))
        merged = merge_constraints(constraints)
        mems = [c for c in merged if isinstance(c, MemoryLimit)]
        assert len(mems) == 1
        assert mems[0].amount == 512
        assert mems[0].unit == "MB"

    def test_merge_time_takes_minimum(self):
        from lackpy.sandbox.constraints import TimeLimit, merge_constraints
        constraints = (TimeLimit(seconds=120), TimeLimit(seconds=60))
        merged = merge_constraints(constraints)
        times = [c for c in merged if isinstance(c, TimeLimit)]
        assert len(times) == 1
        assert times[0].seconds == 60

    def test_merge_pid_takes_minimum(self):
        from lackpy.sandbox.constraints import PidLimit, merge_constraints
        constraints = (PidLimit(max_pids=32), PidLimit(max_pids=16))
        merged = merge_constraints(constraints)
        pids = [c for c in merged if isinstance(c, PidLimit)]
        assert len(pids) == 1
        assert pids[0].max_pids == 16

    def test_merge_network_restrictive_wins(self):
        from lackpy.sandbox.constraints import NetworkAccess, merge_constraints
        constraints = (NetworkAccess(allow=True), NetworkAccess(allow=False))
        merged = merge_constraints(constraints)
        nets = [c for c in merged if isinstance(c, NetworkAccess)]
        assert len(nets) == 1
        assert nets[0].allow is False

    def test_merge_filesystem_union(self):
        from lackpy.sandbox.constraints import FilesystemMount, merge_constraints
        constraints = (
            FilesystemMount(path="/a", writable=True),
            FilesystemMount(path="/b", writable=False),
        )
        merged = merge_constraints(constraints)
        mounts = [c for c in merged if isinstance(c, FilesystemMount)]
        paths = {m.path for m in mounts}
        assert paths == {"/a", "/b"}

    def test_merge_filesystem_same_path_readonly_wins(self):
        from lackpy.sandbox.constraints import FilesystemMount, merge_constraints
        constraints = (
            FilesystemMount(path="/data", writable=True),
            FilesystemMount(path="/data", writable=False),
        )
        merged = merge_constraints(constraints)
        mounts = [c for c in merged if isinstance(c, FilesystemMount)]
        assert len(mounts) == 1
        assert mounts[0].path == "/data"
        assert mounts[0].writable is False

    def test_merge_bridged_policy_false_wins(self):
        from lackpy.sandbox.constraints import BridgedToolPolicy, merge_constraints
        constraints = (
            BridgedToolPolicy(allowed=True, allowed_tools=("a", "b")),
            BridgedToolPolicy(allowed=False),
        )
        merged = merge_constraints(constraints)
        policies = [c for c in merged if isinstance(c, BridgedToolPolicy)]
        assert len(policies) == 1
        assert policies[0].allowed is False

    def test_merge_bridged_policy_intersection(self):
        from lackpy.sandbox.constraints import BridgedToolPolicy, merge_constraints
        constraints = (
            BridgedToolPolicy(allowed=True, allowed_tools=("a", "b", "c")),
            BridgedToolPolicy(allowed=True, allowed_tools=("b", "c", "d")),
        )
        merged = merge_constraints(constraints)
        policies = [c for c in merged if isinstance(c, BridgedToolPolicy)]
        assert len(policies) == 1
        assert policies[0].allowed is True
        assert set(policies[0].allowed_tools) == {"b", "c"}

    def test_merge_preserves_unmerged_types(self):
        from lackpy.sandbox.constraints import (
            SeccompLogging, ReadonlyRoot, merge_constraints,
        )
        constraints = (SeccompLogging(enabled=True), ReadonlyRoot(enabled=True))
        merged = merge_constraints(constraints)
        assert len(merged) == 2

    def test_merge_readonly_root_restrictive_wins(self):
        from lackpy.sandbox.constraints import ReadonlyRoot, merge_constraints
        constraints = (ReadonlyRoot(enabled=False), ReadonlyRoot(enabled=True))
        merged = merge_constraints(constraints)
        roots = [c for c in merged if isinstance(c, ReadonlyRoot)]
        assert len(roots) == 1
        assert roots[0].enabled is True

    def test_merge_empty_returns_empty(self):
        from lackpy.sandbox.constraints import merge_constraints
        assert merge_constraints(()) == ()
