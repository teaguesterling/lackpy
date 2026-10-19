# /// intent
# goal   = "In data/config.json, set version to 1.3.0 and write back as JSON with indent 2."
# reads  = ["data/config.json"]
# writes = ["data/config.json"]
# output = "none"
# ///

import json
from pathlib import Path

path = Path("data/config.json")
text = path.read_text()
data = json.loads(text)
data["version"] = "1.3.0"
new_text = json.dumps(data, indent=2)
path.write_text(new_text)
