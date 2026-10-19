# /// intent
# goal   = "Change 'Status: draft' to 'Status: final' in notes.md, fail if it does not occur exactly once."
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line must occur exactly once"
path.write_text(text.replace("Status: draft", "Status: final"))
