from pathlib import Path
import re

# Read the log file
text = Path("logs/app.log").read_text()

# Count lines with WARN or ERROR
count = 0
for line in text.splitlines():
    if re.search(r'WARN|ERROR', line):
        count += 1

print(count)
