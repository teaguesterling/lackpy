# /// intent
# goal   = "Count distinct error codes from ERROR lines in logs/app.log"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from collections import Counter
from pathlib import Path

text = Path("logs/app.log").read_text()
codes = Counter()

for line in text.splitlines():
    if "ERROR" in line:
        match = re.search(r"code=([A-Z]\d+)", line)
        if match:
            code = match.group(1)
            codes[code] += 1

for code in sorted(codes):
    print(code, codes[code])
