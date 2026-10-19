# /// intent
# goal   = "Print total number of lines across all files under src"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path


total = 0
paths = Path("src").iterdir()
for p in sorted(paths):
    if p.is_file():
        text = p.read_text()
        line_count = len(text.splitlines())
        total = total + line_count
print(total)
