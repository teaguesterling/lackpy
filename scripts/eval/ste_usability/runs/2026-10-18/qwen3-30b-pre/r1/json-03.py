# /// intent
# goal   = "Print the sum of the size field over all valid JSON rows in data/rows.jsonl, skipping invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

total_size = 0
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    size = row.get("size", 0)
    total_size += size
print(total_size)
