# /// intent
# goal   = "Print every file suffix used under src with the number of files that have it, sorted by suffix."
# reads  = ["src"]
# writes = []
# output = "printed"
# ///

from collections import Counter
from pathlib import Path


def count_suffixes(src_path):
    counts = Counter()
    for p in src_path.rglob("*"):
        if p.is_file():
            suffix = p.suffix
            counts[suffix] += 1
    return counts


src_path = Path("src")
suffixes = count_suffixes(src_path)

for name in sorted(suffixes):
    print(name, suffixes[name])
