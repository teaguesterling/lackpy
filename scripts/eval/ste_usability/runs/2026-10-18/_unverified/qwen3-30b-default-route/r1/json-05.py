# /// intent
# goal   = "Write data/ok.json as a JSON list of IDs from data/rows.jsonl where status is 'ok', skipping invalid lines"
# reads  = ["data/rows.jsonl"]
# writes = ["data/ok.json"]
# output = "printed"          # none | printed | value
# ///

import json
from pathlib import Path

# Read the rows.jsonl file line by line
ok_ids = []
file_path = Path("data/rows.jsonl")

try:
    with file_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse the JSON line
                row = json.loads(line)
                # Check if status is "ok"
                if row.get("status") == "ok":
                    ok_ids.append(row["id"])
            except json.JSONDecodeError:
                # Skip invalid lines
                continue
except FileNotFoundError:
    # If the file does not exist, just use empty list
    pass

# Write the result to data/ok.json
output_path = Path("data/ok.json")
output_path.write_text(json.dumps(ok_ids, indent=2))
