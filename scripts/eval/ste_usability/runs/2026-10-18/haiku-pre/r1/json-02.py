# /// intent
# goal   = "Count rows in data/rows.jsonl by status, skipping invalid JSON, and print counts sorted by status"
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
    counts[row["status"]] += 1
for status in sorted(counts):
    print(status, counts[status])
