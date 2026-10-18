# /// intent
# goal   = "Classify each valid row's size as small, medium or large and print the classes in file order"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path
from collections import OrderedDict

def classify(size):
    """Return 'small', 'medium' or 'large' for the given size."""
    if size < 100:
        return "small"
    elif size < 200:
        return "medium"
    return "large"

def load_rows():
    """Load rows from the JSONL file, skipping invalid lines."""
    path = Path("data/rows.jsonl")
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
            # Ensure required fields exist and size is numeric
            if "id" in row and "size" in row and isinstance(row["size"], (int, float)):
                rows.append(row)
        except json.JSONDecodeError:
            pass
    return rows

def main():
    rows = load_rows()
    for row in rows:
        size = row["size"]
        label = classify(size)
        print(label)

main()
