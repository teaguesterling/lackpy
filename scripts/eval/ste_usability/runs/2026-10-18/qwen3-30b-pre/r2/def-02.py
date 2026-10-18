# /// intent
# goal   = "Load valid JSONL rows from data/rows.jsonl and print the count of valid rows"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
import json
from collections import Counter

def load_valid_rows(path):
    """Load valid JSON lines from a file, skipping bad lines."""
    text = path.read_text()
    valid_rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
            valid_rows.append(row)
        except json.JSONDecodeError:
            continue
    return valid_rows

# Load the valid rows from the JSONL file
path = Path("data/rows.jsonl")
valid_rows = load_valid_rows(path)

# Print the number of valid rows
print(len(valid_rows))
