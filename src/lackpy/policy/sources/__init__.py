"""Policy sources: per-source contributors of tool constraints.

Each module here implements a :class:`~lackpy.policy.layer.PolicySource` that the
:class:`~lackpy.policy.layer.PolicyLayer` resolves in order — the tools source
(static grades/whitelist), the kibitzer source (runtime coaching), and the umwelt
source (mode-aware environment constraints).
"""
