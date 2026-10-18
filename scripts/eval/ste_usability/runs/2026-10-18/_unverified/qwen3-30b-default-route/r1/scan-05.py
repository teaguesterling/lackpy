# /// intent
# goal   = "Write report.txt with one line per .cpp file under src, sorted by name, in the form '<name>: <number of characters>'"
# reads  = ["src/*.cpp"]
# writes = ["report.txt"]
# output = "printed"
# ///

import json
import re
from pathlib import Path

# Read all .cpp files under src directory
src_dir = Path("src")
cpp_files = src_dir.glob("*.cpp")

# Dictionary to store filename and its size
file_sizes = {}

# Process each .cpp file
for file_path in cpp_files:
    # Read the file and count characters
    try:
        content = file_path.read_text()
        file_sizes[file_path.name] = len(content)
    except Exception:
        # Skip any file that cannot be read
        continue

# Sort files by name
sorted_files = sorted(file_sizes.items())

# Write to report.txt
report_path = Path("report.txt")
with report_path.open("w") as f:
    for name, size in sorted_files:
        f.write(f"{name}: {size}\n")
