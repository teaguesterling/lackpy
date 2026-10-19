from pathlib import Path
import json

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs exactly once"
path.write_text(text.replace("Status: draft", "Status: final"))
