# /// intent
# goal   = "Print the sum of the size field over all valid rows in data/rows.jsonl. Skip lines that are not valid JSON."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def load_rows():
    p = Path("data/rows.jsonl")
    text = p.read_text()
    rows = text.splitlines()
    total = 0
    for line in rows:
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += obj.get("size", 0)
    return total

result = load_rows()
print(result)
