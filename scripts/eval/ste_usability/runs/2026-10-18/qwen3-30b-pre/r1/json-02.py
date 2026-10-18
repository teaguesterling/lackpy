# /// intent
# goal   = "Count the rows in data/rows.jsonl by status, skipping lines that are not valid JSON, and print each status and its count sorted by status"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from collections import Counter
from pathlib import Path

counts = Counter()
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "status" in row:
        counts[row["status"]] += 1

for status in sorted(counts):
    print(status, counts[status])
