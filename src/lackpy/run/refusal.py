"""``LackpyRefusal``: what the runtime raises when a program does less than CPython would.

The plain-Python invariant (design §5.3) lets lackpy do less than CPython, never
something different. When the runtime cannot give the CPython answer, it refuses.
``LackpyRefusal`` derives from ``BaseException`` so that a program's
``except Exception`` cannot catch it. It is never bound in a program's
namespace, so a program cannot name it.
"""

from __future__ import annotations

from typing import NoReturn

from ..lang import dictionary


class LackpyRefusal(BaseException):
    """A runtime refusal. ``what`` names the refused operation; ``hint`` names the approved alternative."""

    def __init__(self, what: str, hint: str | None = None) -> None:
        super().__init__(what, hint)
        self.what = what
        self.hint = hint

    def __str__(self) -> str:
        message = f"{self.what} is not allowed in lackpy"
        if self.hint:
            message += f": {self.hint}"
        return message


def refuse(what: str, hint_key: str | None = None) -> NoReturn:
    """Raise a ``LackpyRefusal`` for ``what``, with the ``dictionary.UNAPPROVED`` hint for ``hint_key``."""
    hint = dictionary.UNAPPROVED.get(hint_key) if hint_key is not None else None
    raise LackpyRefusal(what, hint)
