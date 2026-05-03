"""NsjailBackend: compiles SandboxConstraints into nsjail-python config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from nsjail.config import IdMap, MountPt, NsJailConfig
from nsjail.mounts import system_libs, dev_minimal, python_env, proc_mount
from nsjail.presets import apply_cgroup_limits, apply_readonly_root
from nsjail.seccomp import DEFAULT_LOG

from ..backend import CompilationResult, ConstraintWarning, SandboxResult
from ..constraints import (
    CpuLimit,
    FilesystemMount,
    MemoryLimit,
    NetworkAccess,
    NetworkRestriction,
    PidLimit,
    ReadonlyRoot,
    SandboxConstraint,
    SeccompLogging,
    SeccompPolicyConstraint,
    TimeLimit,
    UserMapping,
)


class NsjailBackend:
    name: str = "nsjail"

    def accept_policy_config(self, config: Any) -> CompilationResult | None:
        if isinstance(config, NsJailConfig):
            return CompilationResult(config=config, warnings=[])
        return None

    def compile(
        self,
        constraints: Sequence[SandboxConstraint],
        workspace: Path,
    ) -> CompilationResult:
        cfg = NsJailConfig()
        cfg.mount.extend(system_libs())
        cfg.mount.extend(dev_minimal())
        cfg.mount.extend(python_env())
        cfg.mount.extend(proc_mount())
        warnings: list[ConstraintWarning] = []

        for c in constraints:
            if isinstance(c, MemoryLimit):
                mb = c.amount * 1024 if c.unit == "GB" else c.amount
                apply_cgroup_limits(cfg, memory_mb=mb)
            elif isinstance(c, TimeLimit):
                cfg.time_limit = c.seconds
            elif isinstance(c, PidLimit):
                apply_cgroup_limits(cfg, pids_max=c.max_pids)
            elif isinstance(c, CpuLimit):
                apply_cgroup_limits(cfg, cpu_ms_per_sec=c.ms_per_sec)
            elif isinstance(c, NetworkAccess):
                cfg.clone_newnet = not c.allow
            elif isinstance(c, NetworkRestriction):
                warnings.append(ConstraintWarning(
                    constraint=c,
                    reason="nsjail supports binary network on/off only, not per-destination restrictions",
                ))
            elif isinstance(c, FilesystemMount):
                cfg.mount.append(MountPt(
                    src=c.path, dst=c.path, is_bind=True, rw=c.writable,
                ))
            elif isinstance(c, ReadonlyRoot):
                if c.enabled:
                    apply_readonly_root(cfg)
            elif isinstance(c, SeccompLogging):
                if c.enabled:
                    cfg.seccomp_string.append(str(DEFAULT_LOG))
            elif isinstance(c, SeccompPolicyConstraint):
                cfg.seccomp_string.append(c.policy_string)
            elif isinstance(c, UserMapping):
                cfg.uidmap.append(IdMap(
                    inside_id=str(c.inside_id),
                    outside_id=str(c.outside_id),
                    count=c.count,
                ))
            else:
                warnings.append(ConstraintWarning(
                    constraint=c,
                    reason=f"nsjail backend does not support {type(c).__name__}",
                ))

        return CompilationResult(config=cfg, warnings=warnings)

    async def run(self, config: Any, command: list[str]) -> SandboxResult:
        from nsjail.config import Exe
        from nsjail.runner import Runner, NsJailResult
        cfg: NsJailConfig = config
        cfg.exec_bin = Exe(path=command[0], arg=command[1:])
        runner = Runner()
        result: NsJailResult = await runner.async_run(overrides=cfg)
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            oom_killed=result.oom_killed,
        )
