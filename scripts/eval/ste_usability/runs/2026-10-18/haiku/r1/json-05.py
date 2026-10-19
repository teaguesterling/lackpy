# /// intent
# goal   = "Write a JSON list of ids of rows with status ok to data/ok.json"
# reads  = ["data/rows.jsonl"]
# writes = ["data/ok.json"]
# output = "none"
# ///

import json
from pathlib import Path

p = Path('data/rows.jsonl')
text = p.read_text()
lines = text.split('\n')

ok_ids = []
for line in lines:
    if line.strip():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row['status']
        if status == 'ok':
            id_val = row['id']
            ok_ids.append(id_val)

out_path = Path('data/ok.json')
out_text = json.dumps(ok_ids)
out_path.write_text(out_text)
