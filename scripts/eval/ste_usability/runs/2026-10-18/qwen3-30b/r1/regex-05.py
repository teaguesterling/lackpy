# /// intent
# goal   = "Print the number of lines in logs/app.log whose level is WARN or ERROR."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

# Read the log file
log_path = Path("logs/app.log")
text = log_path.read_text()

# Split into lines and count lines with WARN or ERROR
count = 0
for line in text.splitlines():
    if re.search(r'WARN|ERROR', line):
        count += 1

print(count)
