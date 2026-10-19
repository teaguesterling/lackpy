# /// intent
# goal   = "Print the first tag of each valid JSON row in data/rows.jsonl, or '-' if no tags, skipping invalid lines"
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
    tags = row.get("tags", [])
    if tags:
        print(tags[0])
    else:
        print("-")
