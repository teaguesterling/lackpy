from pathlib import Path
import json

data = Path("data/config.json").read_text()
config = json.loads(data)
owner = config.get("owner", "none")
print(owner)
