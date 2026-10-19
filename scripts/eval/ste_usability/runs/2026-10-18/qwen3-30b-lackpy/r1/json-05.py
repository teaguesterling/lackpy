import json
from pathlib import Path

# Read the JSONL file
text = Path("data/rows.jsonl").read_text()

# Parse each line, skip invalid ones
ids = []
for line in text.splitlines():
    try:
        row = json.loads(line)
        if row.get("status") == "ok":
            ids.append(row["id"])
    except json.JSONDecodeError:
        continue

# Write the result to ok.json
Path("data/ok.json").write_text(json.dumps(ids))
