import json
from collections import Counter
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
