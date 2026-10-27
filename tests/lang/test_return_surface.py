"""Return-surface test (spec section 5.1, stepy Plan 3 Task 2).

A member of the dictionary hands the program an object. The program may then
reach every public attribute of that object. The validator bans underscore
names and ``DENIED_ATTRIBUTES``, so every other public attribute is open.
This test lists the types that the approved members return, computes the real
public attributes of each, and asserts that every one is either denied by the
static rules or named in ``SURFACES`` as harmless.

Harmless means data, a pure method, or a method that returns another type in
this table. A new public attribute (for example from a Python upgrade) fails
the test by name until someone classifies it. A route to a module, a frame, a
file or a namespace must never be listed here: report it and deny it by name.
"""

import csv
import difflib
import itertools
import xml.etree.ElementTree as ET

import pytest

from lackpy.lang.validator import validate
from lackpy.run.paths import PathPolicy, make_path_class

_GEN = {"close", "send", "throw", "gi_running", "gi_suspended", "gi_yieldfrom"}  # gi_frame, gi_code: denied
_DICT = {"clear", "copy", "fromkeys", "get", "items", "keys", "pop", "popitem", "setdefault", "update", "values"}
_LIST = {"append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove", "reverse", "sort"}
_DIALECT = {"delimiter", "doublequote", "escapechar", "lineterminator", "quotechar", "quoting",
            "skipinitialspace", "strict"}  # all data


def _stat(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    return make_path_class(PathPolicy(tmp_path, []))("a.txt").stat()


def _dict_reader():
    return csv.DictReader(["a,b", "1,2"])


def _grouper():
    return next(itertools.groupby([1, 1, 2]))[1]


# label -> (factory taking tmp_path, public names a program may reach and that are harmless)
SURFACES = {
    "csv.reader": (lambda t: csv.reader(["a,b"]), {"dialect", "line_num"}),
    "csv.reader.dialect": (lambda t: csv.reader(["a,b"]).dialect, _DIALECT),
    # fieldnames: list of str. reader: a csv.reader (in this table). dialect: a str unless the program passes one.
    "csv.DictReader": (lambda t: _dict_reader(), {"dialect", "fieldnames", "line_num", "reader", "restkey", "restval"}),
    "csv.DictReader row": (lambda t: next(_dict_reader()), _DICT),
    # find/findall/iterfind/iter/itertext/makeelement return an Element, a generator or a list of them.
    "ElementTree.Element": (lambda t: ET.fromstring("<a x='1'><b/></a>"), {
        "append", "attrib", "clear", "extend", "find", "findall", "findtext", "get", "insert", "items",
        "iter", "iterfind", "itertext", "keys", "makeelement", "remove", "set", "tag", "tail", "text"}),
    "ElementTree.Element.iter()": (lambda t: ET.fromstring("<a><b/></a>").iter(), set()),
    "ElementTree.Element.iterfind()": (lambda t: ET.fromstring("<a><b/></a>").iterfind("b"), _GEN),
    "ElementTree.Element.itertext()": (lambda t: ET.fromstring("<a>x</a>").itertext(), set()),
    "difflib.unified_diff": (lambda t: difflib.unified_diff(["a"], ["b"]), _GEN),
    "difflib.get_close_matches": (lambda t: difflib.get_close_matches("a", ["a"]), _LIST),
    "itertools.groupby": (lambda t: itertools.groupby([1, 1, 2]), set()),
    "itertools.groupby group": (lambda t: _grouper(), set()),
    "itertools.chain": (lambda t: itertools.chain([1]), {"from_iterable"}),  # a classmethod that returns a chain
    "itertools.islice": (lambda t: itertools.islice([1], 1), set()),
    "itertools.pairwise": (lambda t: itertools.pairwise([1, 2]), set()),
    "itertools.batched": (lambda t: itertools.batched([1], 1), set()),
    "itertools.product": (lambda t: itertools.product([1]), set()),
    "itertools.combinations": (lambda t: itertools.combinations([1], 1), set()),
    "itertools.accumulate": (lambda t: itertools.accumulate([1]), set()),
    "Path.stat": (_stat, {"st_size", "st_mtime"}),
}


def _denied(name: str) -> bool:
    """Whether the validator rejects ``.name`` (the approach of tests/lang/test_sandbox_escape.py)."""
    result = validate(f"v = [].{name}")
    return not result.valid and any("attribute" in e.lower() and f"'.{name}'" in e for e in result.errors)


@pytest.mark.parametrize("label", SURFACES)
def test_return_surface(label, tmp_path):
    if label == "itertools.batched" and not hasattr(itertools, "batched"):
        pytest.skip("itertools.batched needs Python 3.12")
    make, harmless = SURFACES[label]
    public = {n for n in dir(make(tmp_path)) if not n.startswith("_")}
    unclassified = sorted(n for n in public - harmless if not _denied(n))
    assert not unclassified, f"{label}: unclassified public attributes {unclassified}"
    stale = sorted(harmless - public)
    assert not stale, f"{label}: listed as harmless but not an attribute {stale}"
    for name in sorted(harmless):
        assert not _denied(name), f"{label}: .{name} is listed as harmless but is denied"
