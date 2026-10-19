# /// intent
# goal   = "Print the total number of lines across all files under src."
# reads  = ["data/config.json", "data/rows.jsonl", "logs/app.log", "notes.md", "registry.cpp", "src/a.cpp", "src/b.cpp", "src/c.cpp", "src/util.h"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def line_count(text):
    return text.count('\n') + 1

paths = Path("src").iterdir()
total = 0
for p in sorted(paths):
    if p.is_file():
        text = p.read_text()
        total = total + line_count(text)

print(total)
