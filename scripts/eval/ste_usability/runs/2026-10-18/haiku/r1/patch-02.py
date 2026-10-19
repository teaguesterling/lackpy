# /// intent
# goal   = "Change Status line from draft to final and fail if it doesn't occur exactly once"
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

from pathlib import Path

p = Path('notes.md')
text = p.read_text()
old = 'Status: draft'
new = 'Status: final'
count = text.count(old)
assert count == 1, f"Expected exactly 1 occurrence, found {count}"
updated = text.replace(old, new)
p.write_text(updated)
