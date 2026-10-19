from pathlib import Path
import re
from datetime import datetime

# Read the log file
text = Path("logs/app.log").read_text()

# Extract timestamps from each line using regex
# Assuming format like: 2023-10-05 14:30:25.123
timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

# Find all timestamps
timestamps = []
for line in text.splitlines():
    match = timestamp_pattern.search(line)
    if match:
        timestamps.append(match.group(1))

# Check if we have any timestamps
if not timestamps:
    print("no timestamps found")
else:
    # Get first and last timestamp
    first = timestamps[0]
    last = timestamps[-1]
    
    # Print first and last timestamp separated by space
    print(f"{first} {last}")
