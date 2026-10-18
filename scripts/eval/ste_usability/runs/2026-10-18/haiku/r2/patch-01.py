# /// intent
# goal   = "Replace docx note with shorter version and fail if original doesn't occur exactly once"
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

from pathlib import Path

p = Path('registry.cpp')
text = p.read_text()
old = 'images and footnotes are not read yet'
new = 'footnotes are not read yet'
count = text.count(old)
assert count == 1, f"Expected exactly 1 occurrence, found {count}"
updated = text.replace(old, new)
p.write_text(updated)
