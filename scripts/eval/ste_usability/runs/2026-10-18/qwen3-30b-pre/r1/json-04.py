# /// intent
# goal   = "Print the IDs of rows in data/rows.jsonl with status 'fail', one per line, in file order, skipping invalid lines."
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
        continue
    if row["status"] == "fail":
        print(row["id"])
