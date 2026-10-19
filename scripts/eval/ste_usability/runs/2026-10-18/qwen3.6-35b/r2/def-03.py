# /// intent
# goal   = "Classify each valid row's size as small, medium or large and print the classes in file order"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def classify(size):
    if size < 100:
        return "small"
    if size < 200:
        return "medium"
    return "large"


def load_rows():
    path = Path("data/rows.jsonl")
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
            if "id" in row and "size" in row and isinstance(row["size"], int):
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
