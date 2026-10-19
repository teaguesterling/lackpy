from pathlib import Path
from collections import Counter

counts = Counter()
paths = Path("src").rglob("*")
for p in paths:
    if p.is_file():
        suffix = p.suffix
        counts[suffix] += 1

for suffix in sorted(counts):
    print(suffix, counts[suffix])
