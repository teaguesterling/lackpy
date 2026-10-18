# /// intent
# goal   = "Print the first tag of each valid row in data/rows.jsonl, or '-' when the row has no tags. Skip invalid lines."
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
    tags = row["tags"]
    if tags:
        print(tags[0])
    else:
        print("-")
