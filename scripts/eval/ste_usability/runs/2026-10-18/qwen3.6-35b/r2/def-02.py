# /// intent
# goal   = "Define a function that loads the valid rows of a JSONL text, skipping bad lines, and use it to print how many valid rows data/rows.jsonl has."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def load_rows(text):
    rows = []
    for line in text.split('\n'):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


text = Path('data/rows.jsonl').read_text()
rows = load_rows(text)
print(len(rows))
