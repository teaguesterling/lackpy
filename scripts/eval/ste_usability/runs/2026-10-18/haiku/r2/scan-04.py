# /// intent
# goal   = "Print every file suffix used under src with the count of files, sorted by suffix"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
from collections import Counter

src_dir = Path('src')
suffixes = []

for file_path in src_dir.iterdir():
    if file_path.is_file():
        suffix = file_path.suffix
        suffixes.append(suffix)

counts = Counter(suffixes)

for suffix in sorted(counts):
    print(f"{suffix} {counts[suffix]}")
