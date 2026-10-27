"""Imports at run time: proxies for the dictionary modules, refusals for the rest.

This replaces ``test_import_fail_closed.py``, which pinned that imports failed
closed until the proxies existed (design §8, gate item 9).

The four alias escape routes of design §5.1 are run through the runner with
the validator bypassed: the static check cannot follow an alias, so the proxy
must close each route by itself. The validator rejects every one of these
programs, so the full validated path cannot run them; the last class proves
that rejection, and runs the programs it does admit through the service.
"""

import pathlib
import sys

import pytest

from lackpy.lang import validate
from lackpy.run.runner import RestrictedRunner
from lackpy.service import LackpyService


def _run(program):
    return RestrictedRunner().run(program)


# Each escape: (module, the non-member attribute that names another module).
ESCAPES = [
    ("statistics", "sys"),
    ("json", "codecs"),
    ("re", "functools"),
    ("collections", "abc"),
]


def _escape_programs():
    for module, attr in ESCAPES:
        yield f"import {module}\n{module}.{attr}"
        yield f"import {module}\nm = {module}\nm.{attr}"
        yield f"import {module} as m\nm.{attr}"
        yield f"import {module}\nm = [{module}][0]\nm.{attr}"
        yield f"from {module} import {attr}\n{attr}"
        yield f"from {module} import {attr} as z\nz"


ESCAPE_PROGRAMS = list(_escape_programs())

HANDLERS = ["ValueError", "KeyError", "OSError", "Exception", "(ValueError, KeyError, OSError)"]


def _wrapped(program, handler):
    body = "\n".join("    " + line for line in program.splitlines())
    return f"r = 0\ntry:\n{body}\nexcept {handler}:\n    r = 1\nr"


class TestEscapesRefuseAtRunTime:
    @pytest.mark.parametrize("program", ESCAPE_PROGRAMS)
    def test_escape_refuses(self, program):
        result = _run(program)
        assert not result.success
        assert "not allowed in lackpy" in (result.error or ""), result.error

    @pytest.mark.parametrize("handler", HANDLERS)
    @pytest.mark.parametrize("program", ESCAPE_PROGRAMS)
    def test_escape_refusal_is_not_caught(self, program, handler):
        # Review Focus 3: a handler the validator allows must not swallow it.
        result = _run(_wrapped(program, handler))
        assert not result.success
        assert "not allowed in lackpy" in (result.error or ""), result.error


class TestImports:
    def test_import_os_refuses_with_hint(self):
        result = _run("import os\nos.getcwd()")
        assert not result.success
        assert "use pathlib.Path" in (result.error or ""), result.error

    @pytest.mark.parametrize("handler", HANDLERS)
    def test_import_os_refusal_is_not_caught(self, handler):
        result = _run(_wrapped("import os", handler))
        assert not result.success
        assert "use pathlib.Path" in (result.error or ""), result.error

    @pytest.mark.parametrize("program", [
        "import pathlib\npathlib",
        "from pathlib import Path\nPath('x')",
        "import pathlib as p\np.Path('x')",
    ])
    def test_pathlib_refuses_until_the_path_policy(self, program):
        # Ruling C: the real pathlib module or Path class never reaches a program.
        result = _run(program)
        assert not result.success
        assert "Path arrives with the PathPolicy" in (result.error or ""), result.error

    def test_relative_import_refuses(self):
        result = _run("from . import json\njson")
        assert not result.success
        assert "not allowed in lackpy" in (result.error or ""), result.error

    def test_json_loads(self):
        result = _run("import json\njson.loads('[1]') == [1]")
        assert result.success, result.error
        assert result.output is True

    def test_from_collections_import_counter(self):
        result = _run("from collections import Counter\nc = Counter('abca')\nc['a']")
        assert result.success, result.error
        assert result.output == 2

    def test_counter_repr_matches_plain_python(self):
        result = _run("from collections import Counter\nrepr(Counter('ab'))")
        assert result.success, result.error
        assert result.output == "Counter({'a': 1, 'b': 1})"

    @pytest.mark.parametrize("handler", HANDLERS)
    def test_working_import_inside_a_handler(self, handler):
        result = _run(_wrapped("import json\nx = json.loads('[1]')", handler) + "\nx")
        assert result.success, result.error
        assert result.output == [1]

    def test_setattr_on_a_module_refuses(self):
        result = _run("import json\nj = json\nj.loads = len")
        assert not result.success
        assert "not allowed in lackpy" in (result.error or ""), result.error


@pytest.fixture
def service(tmp_path):
    return LackpyService(workspace=tmp_path)


class TestValidatedPath:
    @pytest.mark.parametrize("program", ESCAPE_PROGRAMS)
    def test_validator_rejects_every_escape(self, program):
        assert not validate(program).valid

    @pytest.mark.asyncio
    @pytest.mark.parametrize("program", ESCAPE_PROGRAMS)
    async def test_service_does_not_run_an_escape(self, service, program):
        result = await service.run_program(program)
        assert not result.success

    @pytest.mark.asyncio
    async def test_service_runs_json_loads(self, service):
        result = await service.run_program("import json\njson.loads('[1]') == [1]")
        assert result.success, result.error
        assert result.output is True

    @pytest.mark.asyncio
    async def test_service_runs_from_collections_import(self, service):
        result = await service.run_program(
            "from collections import Counter\nCounter('abca')['a']")
        assert result.success, result.error
        assert result.output == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("program", ["import pathlib\npathlib.Path('x')", "from pathlib import Path\nPath('x')"])
    async def test_service_serves_the_path_stand_in(self, service, program):
        # Retired (ruling C, until Task 7): the service had no PathPolicy, so pathlib
        # refused with "Path arrives with the PathPolicy". Task 7 wires load_policy.
        assert validate(program).valid
        result = await service.run_program(program)
        assert result.success, result.error
        assert result.output == repr(pathlib.Path("x"))


class TestNewModuleMembers:
    def test_new_module_members_run(self):
        result = _run('import xml.etree.ElementTree as ET\nET.fromstring("<a><b/></a>").find("b").tag')
        assert result.success, result.error
        assert result.output == "b"
        result = _run('from xml.etree.ElementTree import fromstring\nfromstring("<a><b/></a>").find("b").tag')
        assert result.success, result.error
        assert result.output == "b"

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="itertools.batched needs Python 3.12")
    def test_batched_runs(self):
        result = _run("import itertools\nlist(itertools.batched([1, 2, 3], 2))")
        assert result.success, result.error
        assert result.output == [(1, 2), (3,)]

    @pytest.mark.parametrize("program", [
        "import xml\nxml",
        "import xml.etree.ElementTree as ET\nET.sys",
        "import xml.etree.ElementTree\nxml.etree.sys",
        "import xml.etree as E\nE",
        "import xml.etree.ElementTree\nxml.sys",
    ])
    def test_dotted_imports_stay_closed(self, program):
        assert not _run(program).success
