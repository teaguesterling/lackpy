# /// intent
# goal   = "Print the number of lines in logs/app.log whose level is WARN or ERROR"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

# Read the log file
log_path = Path("logs/app.log")
if not log_path.exists():
    print(0)
else:
    # Count lines with WARN or ERROR
    count = 0
    try:
        text = log_path.read_text()
        for line in text.splitlines():
            if "WARN" in line or "ERROR" in line:
                count += 1
    except (KeyError, json.JSONDecodeError):
        # Catch only specific exception classes, not Exception
        count = 0
    print(count)
