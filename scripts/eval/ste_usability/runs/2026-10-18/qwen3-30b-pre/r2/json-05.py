# /// intent
# goal   = "Write a JSON list of the IDs of valid rows in data/rows.jsonl that have status 'ok', skipping invalid lines"
# reads  = ["data/rows.jsonl"]
# writes = ["data/ok.json"]
# output = "none"
# ///

import json
from pathlib import Path

# Read the JSONL file line by line
text = Path("data/rows.jsonl").read_text()

# List to store IDs of rows with status "ok"
ok_ids = []

# Process each line
for line in text.splitlines():
    try:
        # Parse the line as JSON
        row = json.loads(line)
        # Check if the status is "ok"
        if row["status"] == "ok":
            ok_ids.append(row["id"])
    except (json.JSONDecodeError, KeyError, TypeError):
        # Skip invalid lines or lines missing expected keys
        continue

# Write the list of IDs to data/ok.json
Path("data/ok.json").write_text(json.dumps(ok_ids))
