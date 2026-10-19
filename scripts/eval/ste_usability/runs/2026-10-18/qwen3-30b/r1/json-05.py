# /// intent
# goal   = "Write data/ok.json as a JSON list of IDs from data/rows.jsonl where status is 'ok', skipping invalid lines"
# reads  = ["data/rows.jsonl"]
# writes = ["data/ok.json"]
# output = "printed"
# ///

import json
from pathlib import Path

# Read the rows.jsonl file line by line
ok_ids = []
data_path = Path("data/rows.jsonl")

try:
    with data_path.open('r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse the JSON line
                entry = json.loads(line)
                # Check if status is 'ok'
                if entry.get("status") == "ok":
                    ok_ids.append(entry["id"])
            except (json.JSONDecodeError, TypeError, KeyError):
                # Skip invalid lines
                continue
except FileNotFoundError:
    # If the file doesn't exist, write empty list
    pass

# Write the result to data/ok.json
output_path = Path("data/ok.json")
output_path.write_text(json.dumps(ok_ids, indent=2))
