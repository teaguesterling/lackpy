# /// intent
# goal   = "Print each distinct error code from the ERROR lines of logs/app.log with how often it occurs, sorted by code."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import json
import re
import collections

def main():
    try:
        text = Path('logs/app.log').read_text()
    except FileNotFoundError:
        return

    lines = text.split('\n')
    counts = collections.Counter()

    for line in lines:
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        level = parts[1]
        if level != 'ERROR':
            continue
        match = re.search(r'code=([A-Z]\d+)', line)
        if match:
            code = match.group(1)
            counts[code] += 1

    for code in sorted(counts.keys()):
        print(code, counts[code])

main()
