# /// intent
# goal   = "Load valid rows from data/rows.jsonl, skipping invalid lines, and print the count of valid rows"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def load_valid_rows(path: Path) -> list:
    """Load valid JSON lines from the given file path, skipping invalid lines."""
    valid_rows = []
    try:
        with path.open('r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    valid_rows.append(row)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass  # silently skip any I/O issues
    return valid_rows

# Load valid rows from the JSONL file
rows_path = Path("data/rows.jsonl")
valid_rows = load_valid_rows(rows_path)

# Print the count of valid rows
print(len(valid_rows))
