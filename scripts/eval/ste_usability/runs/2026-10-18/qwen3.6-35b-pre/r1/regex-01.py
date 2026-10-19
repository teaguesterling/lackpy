# /// intent
# goal   = "Print each distinct error code from the ERROR lines of logs/app.log with how often it occurs, sorted by code."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from collections import Counter
from pathlib import Path

text = Path("logs/app.log").read_text()
pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ERROR code=(\S+)")
codes = Counter()
for line in text.splitlines():
    match = pattern.search(line)
    if match is not None:
        codes[match.group(1)] += 1
for code in sorted(codes):
    print(code, codes[code])
