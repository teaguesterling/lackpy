# /// intent
# goal   = "Print the time of the first and last line of logs/app.log"
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

text = Path('logs/app.log').read_text()
lines = text.strip().split('\n')

first_line = lines[0]
last_line = lines[-1]

pattern = r'\d{2}:\d{2}:\d{2}'
first_time = re.search(pattern, first_line).group()
last_time = re.search(pattern, last_line).group()

print(f'{first_time} {last_time}')
