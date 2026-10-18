# /// intent
# goal   = "Replace the docx footnotes note in registry.cpp, failing if it doesn't occur exactly once."
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

from pathlib import Path

path = Path("registry.cpp")
text = path.read_text()
old_note = "images and footnotes are not read yet"
count = text.count(old_note)
assert count == 1, "the docx note occurs exactly once"
new_text = text.replace(old_note, "footnotes are not read yet")
path.write_text(new_text)
