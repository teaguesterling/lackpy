# /// intent
# goal   = "Classify each valid row's size in data/rows.jsonl as small, medium, or large, and print the classification in file order."
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
import json
from collections import Counter

def classify_size(size):
    # Use dict lookup instead of if/elif chain
    return {
        size < 100: "small",
        size < 200: "medium",
        True: "large"
    }[True]

# Read the JSONL file
text = Path("data/rows.jsonl").read_text()

# Process each line
for line in text.splitlines():
    try:
        row = json.loads(line)
        size = row["size"]
        classification = classify_size(size)
        print(classification)
    except (json.JSONDecodeError, KeyError, TypeError):
        continue
