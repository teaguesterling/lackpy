# /// intent
# goal   = "Extract and print function names from .cpp files in src/"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

names = []
paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    for line in text.splitlines():
        if line.startswith("int "):
            match = re.search(r"int\s+(\w+)\s*\(", line)
            if match:
                name = match.group(1)
                names.append(name)

for name in sorted(names):
    print(name)
