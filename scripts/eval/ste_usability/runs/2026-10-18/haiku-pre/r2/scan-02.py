# /// intent
# goal   = "Print names of files under src containing no TODO, sorted"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path


paths = Path("src").iterdir()
for p in sorted(paths):
    if p.is_file():
        text = p.read_text()
        if text.count("TODO") == 0:
            print(p.name)
