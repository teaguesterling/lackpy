# /// intent
# goal   = "Print the time of the first and last line of logs/app.log"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

text = Path("logs/app.log").read_text()
lines = text.splitlines()

first_line = lines[0]
last_line = lines[-1]

match_first = re.search(r"(\d{2}:\d{2}:\d{2})", first_line)
match_last = re.search(r"(\d{2}:\d{2}:\d{2})", last_line)

first_time = match_first.group(1)
last_time = match_last.group(1)

print(first_time, last_time)
