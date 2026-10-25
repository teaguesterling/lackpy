"""The dictionary: the approved modules, their approved members, and the unapproved words.

A program may import only a module in ``MODULES``, and may use only the members
that ``MODULES`` lists for it. The member sets come from the corpus counts in
the plan's research. ``safety.import`` and ``safety.module-attr`` read them.

``PATH_MEMBERS`` and ``PATH_IO_METHODS`` describe a ``Path`` instance. The
validator does not check them yet. The runtime proxy (Plan 2) enforces them.

``UNAPPROVED`` maps a word that the language does not approve to a hint that
names the approved replacement. A violation whose subject is in it carries
that hint.
"""

from __future__ import annotations

MODULES: dict[str, frozenset[str]] = {
    "pathlib": frozenset({"Path"}),
    "json": frozenset({"loads", "dumps", "JSONDecodeError"}),
    "re": frozenset({
        "sub", "subn", "findall", "finditer", "search", "match", "fullmatch", "split",
        "compile", "escape", "error",
        "S", "I", "M", "X", "DOTALL", "IGNORECASE", "MULTILINE", "VERBOSE",
    }),
    "collections": frozenset({"Counter", "defaultdict", "deque"}),
    "math": frozenset({
        "comb", "sqrt", "exp", "floor", "ceil", "log", "log10", "isclose", "isfinite",
        "prod", "gcd", "fsum", "inf", "nan",
    }),
    "statistics": frozenset({"median", "mean", "stdev", "pstdev", "quantiles"}),
    "hashlib": frozenset({"sha256", "sha1"}),
    "datetime": frozenset({"datetime", "date", "timedelta", "timezone", "UTC"}),
    # Plan 3 core vocabulary. A dotted name is the import name; the program
    # writes ``import xml.etree.ElementTree as ET`` (the ``as`` is required).
    "tomllib": frozenset({"loads", "TOMLDecodeError"}),
    "csv": frozenset({"reader", "DictReader", "Error"}),
    "xml.etree.ElementTree": frozenset({"fromstring", "ParseError"}),
    "textwrap": frozenset({"dedent", "indent", "fill"}),
    "html": frozenset({"unescape", "escape"}),
    "unicodedata": frozenset({"normalize"}),
    "difflib": frozenset({"unified_diff", "get_close_matches"}),
    "base64": frozenset({"b64encode", "b64decode"}),
    "itertools": frozenset({
        "groupby", "chain", "islice", "pairwise", "batched", "product", "combinations", "accumulate",
    }),
}

# The oldest Python that has a module (member None) or a member. The validator
# reports ``safety.module-attr`` "<module>.<member> needs Python <version>"
# on an older Python, instead of a member the runtime lacks.
MIN_PYTHON: dict[tuple[str, str | None], tuple[int, int]] = {
    ("tomllib", None): (3, 11),
    ("datetime", "UTC"): (3, 11),
    ("itertools", "batched"): (3, 12),
}

PATH_MEMBERS: frozenset[str] = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes",
    "exists", "is_file", "is_dir", "glob", "rglob", "iterdir", "mkdir",
    "name", "stem", "suffix", "parent", "parts",
    "with_suffix", "relative_to", "joinpath",
    "stat", "with_name", "match", "as_posix", "is_symlink",
})

PATH_IO_METHODS: frozenset[str] = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes",
    "glob", "rglob", "iterdir", "mkdir", "exists", "is_file", "is_dir",
    "stat", "is_symlink",
})

UNAPPROVED: dict[str, str] = {
    "open": "use Path(p).read_text() or Path(p).write_text(s)",
    "json.load": "use json.loads(Path(p).read_text())",
    "json.dump": "use Path(p).write_text(json.dumps(obj))",
    "glob": "use Path(d).glob(pattern)",
    "os": "use pathlib.Path",
    "os.path": "use pathlib.Path",
    "map": "use a comprehension",
    "filter": "use a comprehension with if",
    "format": "use an f-string",
    "collections.OrderedDict": "use dict; a dict keeps insertion order",
    "timezone.utc": "use datetime.UTC",
    "itertools.count": "not approved: an infinite iterator; use range(n) or a list",
    "itertools.cycle": "not approved: an infinite iterator; use range(n) or a list",
    "itertools.repeat": "not approved: an infinite iterator; use range(n) or a list",
}


def hint_for(*subjects: str) -> str | None:
    """Return the hint for the first subject in ``UNAPPROVED``, or None."""
    for subject in subjects:
        if subject in UNAPPROVED:
            return UNAPPROVED[subject]
    return None


def approved_imports_line() -> str:
    """The approved modules as prompt text, in ``MODULES`` order: ``pathlib (Path), json, re, ...``."""
    return ", ".join("pathlib (Path)" if name == "pathlib" else name for name in MODULES)
