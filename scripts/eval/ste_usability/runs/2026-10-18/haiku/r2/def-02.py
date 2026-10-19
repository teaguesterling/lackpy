# /// intent
# goal   = "Load valid rows from a JSONL file and count them"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def load_valid_rows(text):
    rows = []
    for line in text.split('\n'):
        if not line:
            continue
        try:
            row = json.loads(line)
            rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows

text = Path('data/rows.jsonl').read_text()
rows = load_valid_rows(text)
print(len(rows))
