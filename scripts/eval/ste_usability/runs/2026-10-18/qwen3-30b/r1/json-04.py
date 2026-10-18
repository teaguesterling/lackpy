# /// intent
# goal   = "Print the IDs of rows in data/rows.jsonl with status 'fail', one per line, in file order, skipping invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the rows.jsonl file
path = Path("data/rows.jsonl")

# Process each line in the file
with path.open('r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        try:
            # Parse the JSON line
            data = json.loads(line)
            # Check if status is "fail"
            if data.get("status") == "fail":
                # Print the ID
                print(data["id"])
        except json.JSONDecodeError:
            # Skip invalid lines
            continue
