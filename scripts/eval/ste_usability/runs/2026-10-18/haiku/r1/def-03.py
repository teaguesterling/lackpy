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
    try:
        row = json.loads(line)
        size = row['size']
        classification = classify_size(size)
        print(classification)
    except json.JSONDecodeError:
        pass
