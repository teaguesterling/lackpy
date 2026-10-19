"""The NOTICE file agrees with the Apache-2.0 LICENSE (final review I1).

Apache-2.0 section 4(d) makes a redistributor carry the NOTICE text, so a
NOTICE that names another license misstates the terms of every copy.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notice_names_no_other_license():
    text = (ROOT / "NOTICE").read_text()
    for stale in ("GNU", "Affero", "AGPL", "COMMERCIAL.md", "Commercial licensing"):
        assert stale not in text


def test_notice_names_the_project_and_its_copyright():
    text = (ROOT / "NOTICE").read_text()
    assert text.splitlines()[0] == "lackpy"
    assert re.search(r"^Copyright 2024-2026 Teague Sterling$", text, re.M)
    assert "Apache License, Version 2.0" in text


def test_files_named_in_notice_exist():
    text = (ROOT / "NOTICE").read_text()
    for name in re.findall(r"\b[\w-]+\.md\b|\bLICENSE\b", text):
        assert (ROOT / name).exists(), name
