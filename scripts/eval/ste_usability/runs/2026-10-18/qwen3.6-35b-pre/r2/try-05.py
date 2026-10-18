# /// intent
# goal   = "Print the first tag of each valid row in data/rows.jsonl, or '-' when the row has no tags. Skip invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def first_tag(row):
    tags = row.get("tags")
    if tags:
        return tags[0]
    return "-"


text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    tag = first_tag(row)
    print(tag)
