import json
from pathlib import Path

path = Path("data/config.json")
text = path.read_text()
assert text, "config file is not empty"
data = json.loads(text)
assert isinstance(data, dict), "config file is not a dict"
owner = data.get("owner", "none")
print(owner)
