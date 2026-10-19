# /// intent
# goal   = "Print the number of lines in logs/app.log whose level is WARN or ERROR."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

text = Path("logs/app.log").read_text()
lines = text.splitlines()
count = 0
for line in lines:
    parts = line.split(None, 3)
    if len(parts) < 3:
        continue
    level = parts[1]
    if level in ("WARN", "ERROR"):
        count += 1
print(count)
