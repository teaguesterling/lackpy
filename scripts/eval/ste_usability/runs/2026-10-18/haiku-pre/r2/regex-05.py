# /// intent
# goal   = "Count lines in logs/app.log with level WARN or ERROR"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

text = Path("logs/app.log").read_text()
count = 0
for line in text.splitlines():
    if "WARN" in line or "ERROR" in line:
        count += 1
print(count)
