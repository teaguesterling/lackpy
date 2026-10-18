# /// intent
# goal   = "Print the total number of lines across all files under the src directory"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Get all .cpp files in src directory
src_dir = Path("src")
cpp_files = src_dir.glob("*.cpp")

# Initialize line counter
total_lines = 0

# Iterate over each .cpp file and count lines
for file_path in cpp_files:
    if file_path.is_file():
        try:
            # Read file content and count lines
            content = file_path.read_text()
            lines = content.splitlines()
            total_lines += len(lines)
        except Exception:
            # Skip any file that cannot be read
            pass

# Print the total number of lines
print(total_lines)
