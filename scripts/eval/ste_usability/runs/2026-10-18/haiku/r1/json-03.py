# /// intent
# goal   = "Print the sum of the size field over all valid rows in data/rows.jsonl"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

p = Path('data/rows.jsonl')
text = p.read_text()
lines = text.split('\n')

total = 0
for line in lines:
    if line.strip():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        size = row['size']
        total = total + size

print(total)
