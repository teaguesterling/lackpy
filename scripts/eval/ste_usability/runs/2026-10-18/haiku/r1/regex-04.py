# /// intent
# goal   = "Extract function names from .cpp files in src directory and print sorted"
# reads  = ["src/"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

src_dir = Path('src')
functions = []

for cpp_file in src_dir.glob('*.cpp'):
    content = cpp_file.read_text()
    lines = content.split('\n')
    for line in lines:
        if line.startswith('int '):
            match = re.search(r'int\s+(\w+)\s*\(', line)
            if match:
                functions.append(match.group(1))

for func in sorted(functions):
    print(func)
