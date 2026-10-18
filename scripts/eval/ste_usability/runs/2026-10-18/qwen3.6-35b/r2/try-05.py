# /// intent
# goal   = "Print the first tag of each valid row in data/rows.jsonl, or '-' when the row has no tags. Skip invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def process_rows():
    try:
        text = Path('data/rows.jsonl').read_text()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tags = row.get('tags', [])
        if tags:
            print(tags[0])
        else:
            print('-')
    return None


process_rows()
