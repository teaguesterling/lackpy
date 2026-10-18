# /// intent
# goal   = "For each line of data/rows.jsonl, print 'bad' if the line is not valid JSON, else the row id."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print("bad")
        continue
    row_id = row["id"]
    print(row_id)
