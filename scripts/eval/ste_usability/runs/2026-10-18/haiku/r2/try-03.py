# /// intent
# goal   = "For each line of data/rows.jsonl, print 'bad' if not valid JSON, else print the row id"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

lines = Path("data/rows.jsonl").read_text().split('\n')
for line in lines:
    if not line:
        continue
    try:
        row = json.loads(line)
        print(row["id"])
    except json.JSONDecodeError:
        print("bad")
        continue
