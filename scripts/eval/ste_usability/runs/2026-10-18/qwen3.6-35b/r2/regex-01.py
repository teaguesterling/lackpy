# /// intent
# goal   = "Print each distinct error code from the ERROR lines of logs/app.log with how often it occurs, sorted by code."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import collections
import re

def read_log():
    from pathlib import Path
    p = Path('logs/app.log')
    text = p.read_text()
    return text

def count_codes(text):
    counts = collections.Counter()
    lines = text.split('\n')
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
    return counts

def main():
    counts = count_codes(read_log())
    for code in sorted(counts.keys()):
        print(code, counts[code])

main()
