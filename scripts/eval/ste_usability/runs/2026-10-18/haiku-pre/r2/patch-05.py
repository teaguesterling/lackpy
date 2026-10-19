# /// intent
# goal   = "In notes.md, add '- release notes' after '- changelog', failing if the latter doesn't occur exactly once."
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("- changelog") == 1, "the changelog line occurs exactly once"
new_text = text.replace("- changelog", "- changelog\n- release notes")
path.write_text(new_text)
