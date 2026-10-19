# /// intent
# goal   = "Print the sum of the size field over all valid rows in data/rows.jsonl"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

total = 0
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    total += row["size"]
print(total)
