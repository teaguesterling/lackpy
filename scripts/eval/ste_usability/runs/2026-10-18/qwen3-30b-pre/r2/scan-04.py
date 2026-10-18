# /// intent
# goal   = "Print every file suffix used under src with the number of files that have it, sorted by suffix"
# reads  = ["src/*.cpp", "src/util.h"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
from collections import Counter

counts = Counter()

# Get all files under src with .cpp or .h extensions
paths = Path("src").glob("*.*")

for p in paths:
    # Only consider files with .cpp or .h suffixes
    if p.suffix in (".cpp", ".h"):
        # Use the suffix as the key (e.g., ".cpp")
        counts[p.suffix] += 1

# Print results sorted by suffix
for suffix in sorted(counts):
    print(suffix, counts[suffix])
