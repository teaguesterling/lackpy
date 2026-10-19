# /// intent
# goal   = "Count valid JSON rows in data/rows.jsonl"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def load_valid_rows(text):
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


text = Path("data/rows.jsonl").read_text()
rows = load_valid_rows(text)
print(len(rows))
