import json
from pathlib import Path

total = 0
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "size" in row:
        total += row["size"]
print(total)
