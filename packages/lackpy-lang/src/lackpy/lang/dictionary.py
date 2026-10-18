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
    "json": frozenset({"loads", "dumps", "JSONDecodeError"}),
    "re": frozenset({
        "sub", "subn", "findall", "finditer", "search", "match", "fullmatch", "split",
        "compile", "escape", "error",
        "S", "I", "M", "X", "DOTALL", "IGNORECASE", "MULTILINE", "VERBOSE",
    }),
    "collections": frozenset({"Counter", "defaultdict", "deque", "OrderedDict"}),
    "math": frozenset({"comb", "sqrt", "exp"}),
    "statistics": frozenset({"median", "mean", "stdev", "pstdev", "quantiles"}),
    "hashlib": frozenset({"sha256", "sha1"}),
    "datetime": frozenset({"datetime", "date", "timedelta", "timezone", "UTC"}),
    "pathlib": frozenset({"Path"}),
}

PATH_MEMBERS: frozenset[str] = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes",
    "exists", "is_file", "is_dir", "glob", "rglob", "iterdir", "mkdir",
    "name", "stem", "suffix", "parent", "parts",
    "with_suffix", "relative_to", "joinpath",
})

PATH_IO_METHODS: frozenset[str] = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes",
    "glob", "rglob", "iterdir", "mkdir", "exists", "is_file", "is_dir",
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
}


def hint_for(*subjects: str) -> str | None:
    """Return the hint for the first subject in ``UNAPPROVED``, or None."""
    for subject in subjects:
        if subject in UNAPPROVED:
            return UNAPPROVED[subject]
    return None
