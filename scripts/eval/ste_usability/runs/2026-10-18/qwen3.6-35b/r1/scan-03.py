# /// intent
# goal   = "Count total lines in files under src directory"
# reads  = ["src"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def count_lines_in_file(p):
    """Count lines in a file. Returns 0 if file does not exist or is not a file."""
    if not p.exists() or not p.is_file():
        return 0
    return len(p.read_text().splitlines())

def total_lines_in_dir(dir_path):
    """Calculate total lines in all files under a directory."""
    if not dir_path.is_dir():
        return 0
    total = 0
    for child in dir_path.iterdir():
        if child.is_file():
            total += count_lines_in_file(child)
        elif child.is_dir():
            total += total_lines_in_dir(child)
    return total

src = Path("src")
result = total_lines_in_dir(src)
print(result)
