# /// intent
# goal   = "For each line of data/rows.jsonl, print 'bad' if the line is not valid JSON, else the row id."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

path = Path("data/rows.jsonl")
text = path.read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print("bad")
        continue
    print(row["id"])
