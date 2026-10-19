from pathlib import Path
import json

path = Path("notes.md")
text = path.read_text()
assert text.count("- changelog") == 1, "the changelog line occurs exactly once"
lines = text.splitlines()
index = lines.index("- changelog")
lines.insert(index + 1, "- release notes")
path.write_text("\n".join(lines))
