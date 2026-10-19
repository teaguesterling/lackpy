# /// intent
# goal   = "Write a JSON list of the ids of rows in data/rows.jsonl with status ok to data/ok.json"
# reads  = ["data/rows.jsonl"]
# writes = ["data/ok.json"]
# output = "none"
# ///

import json
from pathlib import Path

ok_ids = []
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row["status"] == "ok":
        ok_ids.append(row["id"])
result = json.dumps(ok_ids)
Path("data/ok.json").write_text(result)
