# /// intent
# goal   = "Count the rows in data/rows.jsonl by status, skipping invalid JSON lines, and print each status and its count sorted by status"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path
from collections import Counter

# Read the rows.jsonl file
file_path = Path("data/rows.jsonl")

# Count status occurrences
status_counter = Counter()

# Process each line
try:
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                status = data.get("status")
                if status is not None:
                    status_counter[status] += 1
            except json.JSONDecodeError:
                continue  # Skip invalid JSON lines
except KeyError:
    pass  # Handle missing key, but don't exit or raise

# Print results sorted by status
for status in sorted(status_counter.keys()):
    count = status_counter[status]
    print(f"{status}: {count}")
