"""Creation provenance log for Lackey programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class System:
    """System prompt message."""
    content: str
    role: str = field(default="system", init=False)


@dataclass
class User:
    """User intent or correction message."""
    content: str
    role: str = field(default="user", init=False)


@dataclass
class Assistant:
    """Model-generated output with acceptance status."""
    content: str
    accepted: bool = True
    errors: list[str] | None = None
    strategy: str | None = None
    role: str = field(default="assistant", init=False)


@dataclass
class Log:
    """Ordered record of the generation conversation."""
    messages: list[System | User | Assistant] = field(default_factory=list)

    def to_dicts(self) -> list[dict[str, Any]]:
        result = []
        for msg in self.messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if isinstance(msg, Assistant):
                d["accepted"] = msg.accepted
                if msg.errors:
                    d["errors"] = msg.errors
                if msg.strategy:
                    d["strategy"] = msg.strategy
            result.append(d)
        return result
