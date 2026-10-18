# /// intent
# goal   = "Print the total number of lines across all files under the src directory"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

import json
import re
from pathlib import Path

# Count lines in all .cpp files under src directory
total_lines = 0
src_dir = Path('src')

# Check if src directory exists and is a directory
if src_dir.is_dir():
    # Iterate over all .cpp files in src directory
    for cpp_file in src_dir.glob('*.cpp'):
        # Read the file content
        try:
            content = cpp_file.read_text()
            # Count lines in the file
            lines = content.splitlines()
            total_lines += len(lines)
        except Exception:
            # Skip any file that can't be read
            pass

# Print the total number of lines
print(total_lines)
