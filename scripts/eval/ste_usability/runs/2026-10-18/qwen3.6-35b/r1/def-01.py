# /// intent
# goal   = "Count TODO markers in .cpp files under src and print counts sorted by name."
# reads  = ["data/config.json", "data/rows.jsonl", "logs/app.log", "notes.md", "registry.cpp", "src/a.cpp", "src/b.cpp", "src/c.cpp", "src/util.h"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

# Regex for TODO markers. Case-insensitive, matches "TODO" or "TODO:" or "TODO: "
TODO_RE = re.compile(r'\b[Tt][Oo][Dd][Oo]\b')

def count_todos(text):
    """Return the number of TODO markers in text."""
    return len(TODO_RE.findall(text))

def scan_src():
    src = Path('src')
    if not src.is_dir():
        return []
    cpp_files = [f for f in src.iterdir() if f.is_file() and f.suffix == '.cpp']
    return sorted(cpp_files, key=lambda f: f.name)

def main():
    files = scan_src()
    for f in files:
        text = f.read_text()
        n = count_todos(text)
        print(f"{f.name}: {n}")

main()
