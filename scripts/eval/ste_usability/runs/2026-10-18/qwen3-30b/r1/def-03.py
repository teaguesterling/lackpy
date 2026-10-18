# /// intent
# goal   = "Classify each valid row's size from data/rows.jsonl as small, medium, or large, and print the classification in file order."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def classify_size(size):
    if size < 100:
        return "small"
    elif size < 200:
        return "medium"
    else:
        return "large"

# Read and process each line in rows.jsonl
path = Path("data/rows.jsonl")
if not path.exists():
    print("File data/rows.jsonl not found.")
else:
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'size' in data:
                    size = data['size']
                    classification = classify_size(size)
                    print(classification)
            except json.JSONDecodeError:
                continue
