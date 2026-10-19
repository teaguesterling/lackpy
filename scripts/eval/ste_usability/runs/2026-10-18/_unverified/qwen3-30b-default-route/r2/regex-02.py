# /// intent
# goal   = "Print the time (HH:MM:SS) of the first and the last line of logs/app.log, separated by a space."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

# Read the log file
log_path = Path("logs/app.log")
lines = log_path.read_text().splitlines()

# Extract times from first and last lines using regex
time_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})')

first_time = None
last_time = None

for line in lines:
    match = time_pattern.match(line)
    if match:
        if first_time is None:
            first_time = match.group(1)
        last_time = match.group(1)

# Print first and last times separated by a space
print(f"{first_time} {last_time}")
