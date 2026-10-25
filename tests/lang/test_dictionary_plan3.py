"""Plan 3 core vocabulary: the new modules, the one-way changes, the XML entity checks."""

import time

import pytest

from lackpy.lang import validate
from lackpy.lang import dictionary
from lackpy.run.runner import RestrictedRunner

NEW_MODULE_PROGRAMS = [
    "import tomllib\nd = tomllib.loads('a = 1')\n",
    "import csv\nr = csv.reader(['a,b'])\n",
    "import xml.etree.ElementTree as ET\nroot = ET.fromstring('<a/>')\n",
    "from xml.etree.ElementTree import fromstring\nroot = fromstring('<a/>')\n",
    "import textwrap\ns = textwrap.dedent('  a')\n",
    "import html\ns = html.escape('<')\n",
    "import unicodedata\ns = unicodedata.normalize('NFC', 'a')\n",
    "import difflib\nm = difflib.get_close_matches('a', ['ab'])\n",
    "import base64\nb = base64.b64encode(b'a')\n",
    "import itertools\nx = list(itertools.chain([1], [2]))\n",
    "import math\nx = math.floor(1.5) + math.gcd(4, 6)\ny = math.inf\n",
]


@pytest.mark.parametrize("src", NEW_MODULE_PROGRAMS)
def test_new_modules_validate(src):
    result = validate(src)
    assert result.valid is True, [v.message for v in result.violations]


@pytest.mark.parametrize("src", [
    "import tomllib\ntry:\n    d = tomllib.loads('a')\nexcept tomllib.TOMLDecodeError:\n    d = {}\n",
    "import csv\ntry:\n    r = list(csv.reader(['a']))\nexcept csv.Error:\n    r = []\n",
    "import xml.etree.ElementTree as ET\ntry:\n    r = ET.fromstring('<a')\nexcept ET.ParseError:\n    r = None\n",
])
def test_parse_errors_are_catchable(src):
    result = validate(src)
    assert result.valid is True, [v.message for v in result.violations]


def test_bare_xml_and_undotted_xml_import_refuse():
    assert not validate("import xml\n").valid
    assert not validate("import xml.etree.ElementTree\n").valid  # binds ``xml``, not a dictionary name
    assert not validate("from xml.etree import ElementTree\n").valid


def test_ordered_dict_is_unapproved():
    result = validate("import collections\nd = collections.OrderedDict()\n")
    assert not result.valid
    assert any(v.hint == "use dict; a dict keeps insertion order" for v in result.violations)


def test_timezone_utc_hint():
    for src in ("import datetime\nz = datetime.timezone.utc\n",
                "from datetime import timezone\nz = timezone.utc\n"):
        result = validate(src)
        assert not result.valid, src
        assert any(v.hint == "use datetime.UTC" for v in result.violations), src
    assert validate("import datetime\nz = datetime.timezone(datetime.timedelta(hours=-7))\n").valid
    assert validate("import datetime\nz = datetime.UTC\n").valid


def test_infinite_itertools_refused():
    hint = "not approved: an infinite iterator; use range(n) or a list"
    for name in ("count", "cycle", "repeat"):
        result = validate(f"import itertools\nx = itertools.{name}\n")
        assert not result.valid
        assert any(v.hint == hint for v in result.violations), name


def test_regex_both_ways_still_valid():
    for src in ("import re\nf = re.S\n", "import re\nf = re.DOTALL\n",
                "import re\nm = re.compile('a').search('a')\n", "import re\nm = re.search('a', 'a')\n"):
        assert validate(src).valid, src


def test_python_version_table_names_the_needed_version():
    assert dictionary.MIN_PYTHON[("itertools", "batched")] == (3, 12)
    assert dictionary.MIN_PYTHON[("tomllib", None)] == (3, 11)


def test_approved_imports_line_follows_modules_order():
    line = dictionary.approved_imports_line()
    assert line.startswith("pathlib (Path), json, re")
    names = [part.strip().split(" ")[0] for part in line.split(",")]
    assert names == sorted(set(names), key=names.index)
    assert "xml.etree.ElementTree" in names and "itertools" in names


_ET = "import xml.etree.ElementTree as ET\n"


def test_xml_entity_attacks_refused():
    # Pinned so that a Python with an older expat fails loudly (measured: expat 2.6.1).
    lines = ['<!DOCTYPE l [', '<!ENTITY e0 "lol">']
    for i in range(1, 10):
        lines.append(f'<!ENTITY e{i} "' + f"&e{i - 1};" * 10 + '">')
    doc = "\n".join(lines) + "\n]>\n<r>&e9;</r>"
    bomb = _ET + f"try:\n    ET.fromstring({doc!r})\n    out = 'parsed'\nexcept ET.ParseError as e:\n    out = str(e)\nout\n"
    start = time.monotonic()
    result = RestrictedRunner().run(bomb)
    assert time.monotonic() - start < 2
    assert result.success, result.error
    assert "amplification" in result.output

    xxe = ('<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/hostname">]><r>&x;</r>')
    prog = _ET + f"try:\n    r = ET.fromstring({xxe!r})\n    out = 'parsed:' + (r.text or '')\nexcept ET.ParseError as e:\n    out = str(e)\nout\n"
    result = RestrictedRunner().run(prog)
    assert result.success, result.error
    assert "undefined entity" in result.output
    assert not result.output.startswith("parsed")
