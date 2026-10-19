import json
from pathlib import Path

text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
        print(row["id"])
    except (json.JSONDecodeError, KeyError):
        print("bad")
