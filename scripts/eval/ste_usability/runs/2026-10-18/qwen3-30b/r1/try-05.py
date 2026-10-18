# /// intent
# goal   = "Print the first tag of each valid JSON line in data/rows.jsonl, or '-' if no tags, skipping invalid lines."
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
        # Extract tags if present, otherwise use empty list
        tags = data.get("tags", [])
        # Get the first tag or '-' if no tags
        first_tag = tags[0] if tags else '-'
        # Print the result
        print(first_tag)
    except (json.JSONDecodeError, TypeError, IndexError):
        # Skip invalid lines
        continue
