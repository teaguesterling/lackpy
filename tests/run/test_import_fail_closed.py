"""The runner fails closed on imports until Plan 2."""

import pytest

from lackpy.run.runner import RestrictedRunner


@pytest.mark.parametrize("program", [
    "import json\njson",
    "from pathlib import Path\nPath('x')",
])
def test_import_fails_at_runtime(program):
    """Guard: the program's builtins hold no ``__import__``.

    The validator admits dictionary imports now, but the module and Path
    proxies do not exist yet. If ``__import__`` reached the program's builtins
    before Plan 2 lands, a validated import would hand the program the real
    module. Plan 2 replaces this test with tests of the proxies.
    """
    result = RestrictedRunner().run(program, namespace={})
    assert not result.success
    assert "__import__" in (result.error or ""), result.error
