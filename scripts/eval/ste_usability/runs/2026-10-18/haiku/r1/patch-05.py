# /// intent
# goal   = "Add release notes line after changelog and fail if changelog doesn't occur exactly once"
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

from pathlib import Path

p = Path('notes.md')
text = p.read_text()
old = '- changelog'
count = text.count(old)
assert count == 1, f"Expected exactly 1 occurrence, found {count}"
lines = text.split('\n')
result = []
for line in lines:
    result.append(line)
    if line == old:
        result.append('- release notes')
updated = '\n'.join(result)
p.write_text(updated)
