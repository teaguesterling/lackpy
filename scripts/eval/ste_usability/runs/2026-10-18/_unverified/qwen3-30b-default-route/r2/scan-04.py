# /// intent
# goal   = "Print every file suffix used under src with the number of files that have it, sorted by suffix."
# reads  = ["src/"]
# writes = []
# output = "printed"
# ///

import pathlib
from collections import defaultdict

# Get all files in src directory
src_dir = pathlib.Path("src")
files = src_dir.iterdir()

# Count suffixes
suffix_count = defaultdict(int)

for file in files:
    if file.is_file():
        suffix = file.suffix
        suffix_count[suffix] += 1

# Sort by suffix and print
for suffix in sorted(suffix_count):
    print(f"{suffix} {suffix_count[suffix]}")
