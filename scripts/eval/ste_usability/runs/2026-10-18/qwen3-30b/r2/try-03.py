# /// intent
# goal   = "For each line in data/rows.jsonl, print 'bad' if the line is not valid JSON, otherwise print the row id"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

path = Path("data/rows.jsonl")
lines = path.read_text().splitlines()

for line in lines:
    try:
        row = json.loads(line)
        print(row["id"])
    except (json.JSONDecodeError, KeyError):
        print("bad")
