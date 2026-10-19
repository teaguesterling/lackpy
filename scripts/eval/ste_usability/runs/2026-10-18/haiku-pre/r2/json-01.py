# /// intent
# goal   = "Print the name and version from data/config.json, separated by a space"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

text = Path("data/config.json").read_text()
data = json.loads(text)
print(f"{data['name']} {data['version']}")
