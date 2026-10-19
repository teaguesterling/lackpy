# /// intent
# goal   = "Print each distinct error code from ERROR lines in app.log, along with its frequency, sorted by code."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from collections import defaultdict

# Read the log file
log_path = "logs/app.log"
error_pattern = re.compile(r'ERROR code=([A-Z0-9]+)')

# Count error codes
error_count = defaultdict(int)

try:
    with open(log_path, 'r') as file:
        for line in file:
            if 'ERROR' in line:
                match = error_pattern.search(line)
                if match:
                    code = match.group(1)
                    error_count[code] += 1
except FileNotFoundError:
    # If the file doesn't exist, the count will be empty
    pass

# Sort by error code and print results
for code in sorted(error_count.keys()):
    print(f"{code} {error_count[code]}")
