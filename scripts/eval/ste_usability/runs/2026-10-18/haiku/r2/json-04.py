# /// intent
# goal   = "Print the ids of rows with status fail from data/rows.jsonl in file order"
# reads  = ["data/rows.jsonl"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

p = Path('data/rows.jsonl')
text = p.read_text()
lines = text.split('\n')

for line in lines:
    if line.strip():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row['status']
        if status == 'fail':
            id_val = row['id']
            print(id_val)
