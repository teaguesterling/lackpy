# /// intent
# goal   = "Print the ids of rows in data/rows.jsonl whose status is fail, one per line, in file order. Skip invalid lines."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def parse_row(text):
    """Parse a single JSON row, returning None if invalid."""
    row = None
    try:
        row = json.loads(text)
        if isinstance(row, dict):
            return row
    except json.JSONDecodeError:
        pass
    return None

def main():
    """Main entry point to process rows."""
    path = Path("data/rows.jsonl")
    text = path.read_text()
    for line in text.splitlines():
        row = parse_row(line)
        if row is not None and row.get("status") == "fail":
            print(row["id"])

main()
