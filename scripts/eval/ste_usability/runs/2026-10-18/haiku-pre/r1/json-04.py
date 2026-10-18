# /// intent
# goal   = "Print the ids of rows in data/rows.jsonl whose status is fail, one per line"
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
