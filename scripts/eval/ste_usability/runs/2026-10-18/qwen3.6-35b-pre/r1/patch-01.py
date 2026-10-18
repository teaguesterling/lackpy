# /// intent
# goal   = "In registry.cpp, replace the docx note with a new note, failing if the old note does not occur exactly once."
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

from pathlib import Path

path = Path("registry.cpp")
text = path.read_text()
assert text.count("images and footnotes are not read yet") == 1, "the old note occurs exactly once"
new_text = text.replace("images and footnotes are not read yet", "footnotes are not read yet")
path.write_text(new_text)
