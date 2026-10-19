# /// intent
# goal   = "Print the number of lines in logs/app.log whose level is WARN or ERROR"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the log file
log_path = Path("logs/app.log")
if not log_path.exists():
    print(0)
else:
    # Count lines with WARN or ERROR
    count = 0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "WARN" in line or "ERROR" in line:
                    count += 1
    except Exception:
        # In lackpy, this shouldn't happen, but for safety
        count = 0
    print(count)
