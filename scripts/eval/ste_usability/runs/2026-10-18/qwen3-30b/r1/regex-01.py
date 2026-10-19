# /// intent
# goal   = "Print each distinct error code from ERROR lines in logs/app.log along with its frequency, sorted by code."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from collections import Counter

# Read the log file
with open('logs/app.log', 'r') as file:
    lines = file.readlines()

# Extract error codes from lines containing 'ERROR'
error_codes = []
for line in lines:
    if 'ERROR' in line:
        # Use regex to find the error code (e.g., E12, E40)
        match = re.search(r'code=([A-Z]\d+)', line)
        if match:
            error_codes.append(match.group(1))

# Count frequency of each error code
code_counts = Counter(error_codes)

# Sort by error code and print
for code in sorted(code_counts.keys()):
    count = code_counts[code]
    print(f"{code} {count}")
