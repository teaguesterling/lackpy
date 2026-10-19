# /// intent
# goal   = "Classify sizes from valid rows in a JSONL file"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def classify_size(size):
    classes = [("small", 100), ("medium", 200)]
    for class_name, threshold in classes:
        if size < threshold:
            return class_name
    return "large"

text = Path('data/rows.jsonl').read_text()
for line in text.split('\n'):
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    size = row['size']
    classification = classify_size(size)
    print(classification)
