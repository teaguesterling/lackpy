# /// intent
# goal   = "Count and print the number of lines in logs/app.log with WARN or ERROR level"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

text = Path('logs/app.log').read_text()
lines = text.split('\n')

count = 0
for line in lines:
    if 'WARN' in line or 'ERROR' in line:
        count += 1

print(count)
