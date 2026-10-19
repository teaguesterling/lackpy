from pathlib import Path
import json
from collections import Counter

def classify_size(size):
    if size < 100:
        return "small"
    if size < 200:
        return "medium"
    return "large"

counts = Counter()
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    size = row.get("size")
    if size is None:
        continue
    if not isinstance(size, (int, float)):
        continue
    classification = classify_size(size)
    print(classification)
