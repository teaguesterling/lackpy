import json
from pathlib import Path

path = Path("data/rows.jsonl")
text = path.read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("status") == "fail":
        print(row["id"])
