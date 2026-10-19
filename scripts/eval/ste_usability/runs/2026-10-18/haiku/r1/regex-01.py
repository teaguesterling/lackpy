# /// intent
# goal   = "Extract error codes from ERROR lines in logs/app.log and print with counts, sorted by code"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

text = Path('logs/app.log').read_text()
lines = text.split('\n')

codes = {}
for line in lines:
    if 'ERROR' in line:
        match = re.search(r'code=(\w\d+)', line)
        if match:
            code = match.group(1)
            codes[code] = codes.get(code, 0) + 1

for code in sorted(codes):
    print(f'{code} {codes[code]}')
