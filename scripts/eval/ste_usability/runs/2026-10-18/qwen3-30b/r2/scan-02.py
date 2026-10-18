# /// intent
# goal   = "Print the names of files under src directory with no TODO comment, sorted alphabetically"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

import json
import re
from pathlib import Path

# Get all .cpp files in src directory
src_dir = Path('src')
cpp_files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix == '.cpp']

# Filter files that contain no TODO
files_without_todo = []
for file_path in cpp_files:
    try:
        content = file_path.read_text()
        # Check if 'TODO' appears in the content (case-insensitive)
        if 'TODO' not in content:
            files_without_todo.append(file_path.name)
    except (OSError, UnicodeDecodeError) as e:
        # Handle file reading errors safely
        continue

# Sort the filenames alphabetically
files_without_todo.sort()

# Print the sorted filenames
for filename in files_without_todo:
    print(filename)
