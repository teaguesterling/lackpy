# /// intent
# goal   = "Print the total number of lines across all files under the src directory"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

total_lines = 0
src_dir = Path('src')

if src_dir.is_dir():
    for cpp_file in src_dir.glob('*.cpp'):
        try:
            content = cpp_file.read_text()
            lines = content.splitlines()
            total_lines += len(lines)
        except (OSError, UnicodeDecodeError):
            pass

print(total_lines)
