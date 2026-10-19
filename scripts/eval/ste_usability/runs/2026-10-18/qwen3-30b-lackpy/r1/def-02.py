import json
from pathlib import Path

def load_valid_rows(path):
    """Load valid rows from a JSONL file, skipping bad lines."""
    text = Path(path).read_text()
    valid_rows = 0
    for line in text.splitlines():
        try:
            json.loads(line)
            valid_rows += 1
        except json.JSONDecodeError:
            continue
    return valid_rows

count = load_valid_rows("data/rows.jsonl")
print(count)
