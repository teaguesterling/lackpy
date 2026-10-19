# /// intent
# goal   = "Print the ids of rows in data/rows.jsonl whose status is fail, one per line, in file order, skipping invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the rows.jsonl file
path = Path("data/rows.jsonl")
lines = path.read_text().splitlines()

# Process each line
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        # Parse the JSON line
        data = json.loads(line)
        # Check if status is "fail"
        if data.get("status") == "fail":
            print(data["id"])
    except (json.JSONDecodeError, TypeError, KeyError):
        # Skip invalid lines
        continue
