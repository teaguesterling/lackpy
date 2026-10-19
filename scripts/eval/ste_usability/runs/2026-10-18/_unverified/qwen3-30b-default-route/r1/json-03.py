# /// intent
# goal   = "Print the sum of the size field over all valid rows in data/rows.jsonl, skipping lines that are not valid JSON."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

total_size = 0
file_path = Path("data/rows.jsonl")

try:
    with file_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict) and 'size' in row:
                    total_size += row['size']
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    pass

print(total_size)
