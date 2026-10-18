import json
from pathlib import Path

# /// intent
# goal   = "Print the name and version from data/config.json, separated by a space."
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

text = Path("data/config.json").read_text()
data = json.loads(text)
assert isinstance(data["name"], str), "name must be a string"
assert isinstance(data["version"], str), "version must be a string"
print(data["name"], data["version"])
