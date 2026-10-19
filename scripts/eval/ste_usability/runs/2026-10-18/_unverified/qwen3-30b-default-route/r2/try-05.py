# /// intent
# goal   = "Print the first tag of each valid JSON line in data/rows.jsonl, or '-' if no tags, skipping invalid lines"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the rows.jsonl file
path = Path("data/rows.jsonl")

# Process each line in the file
for line in path.read_text().splitlines():
    # Skip empty lines
    if not line.strip():
        continue
    
    try:
        # Parse the JSON line
        row = json.loads(line)
        
        # Extract the tags list
        tags = row.get("tags", [])
        
        # Print the first tag if it exists, otherwise '-'
        if tags:
            print(tags[0])
        else:
            print("-")
            
    except (json.JSONDecodeError, TypeError, KeyError):
        # Skip invalid lines (not valid JSON or missing required fields)
        continue
