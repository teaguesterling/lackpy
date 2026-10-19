# /// intent
# goal   = "Print the first tag of each valid row in data/rows.jsonl, or '-' when no tags"
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
        tags = row.get("tags", [])
        if tags:
            print(tags[0])
        else:
            print("-")
    except json.JSONDecodeError:
        continue
