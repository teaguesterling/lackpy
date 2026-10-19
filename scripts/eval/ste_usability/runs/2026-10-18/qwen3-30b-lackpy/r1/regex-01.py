# /// intent
# goal   = "Print each distinct error code from ERROR lines in logs/app.log with its frequency, sorted by code"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
import re
from collections import Counter

# Read the log file
text = Path("logs/app.log").read_text()

# Extract error codes from lines containing "ERROR"
error_codes = re.findall(r'ERROR.*?([A-Z]\d+)', text)

# Count frequency of each error code
counts = Counter(error_codes)

# Print each error code and its count, sorted by code
for code in sorted(counts):
    print(code, counts[code])
