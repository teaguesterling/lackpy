# /// intent
# goal   = "Print file suffixes under src with count of files, sorted by suffix"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
from collections import Counter


suffixes = Counter()
paths = Path("src").iterdir()
for p in sorted(paths):
    if p.is_file():
        suffix = p.suffix
        suffixes[suffix] += 1
for suffix in sorted(suffixes):
    print(suffix, suffixes[suffix])
