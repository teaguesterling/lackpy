# /// intent
# goal   = "Update the version in data/config.json to 1.3.0 and write it back with 2-space indentation"
# reads  = ["data/config.json"]
# writes = ["data/config.json"]
# output = "none"
# ///

from pathlib import Path
import json

path = Path("data/config.json")
text = path.read_text()
data = json.loads(text)
data["version"] = "1.3.0"
path.write_text(json.dumps(data, indent=2))
