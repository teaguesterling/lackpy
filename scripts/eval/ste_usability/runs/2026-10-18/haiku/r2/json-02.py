# /// intent
# goal   = "Count rows in data/rows.jsonl by status, skipping invalid JSON lines"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path
from collections import Counter

p = Path('data/rows.jsonl')
text = p.read_text()
lines = text.split('\n')

counts = Counter()
for line in lines:
    if line.strip():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row['status']
        counts[status] += 1

statuses = sorted(counts.keys())
for status in statuses:
    print(f'{status} {counts[status]}')
