# /// intent
# goal   = "Print the value of the 'owner' key in data/config.json, or 'none' if missing"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

text = Path("data/config.json").read_text()
config = json.loads(text)
owner = config.get("owner", "none")
print(owner)
