from pathlib import Path
import json

data = Path("data/config.json").read_text()
config = json.loads(data)
print(config["name"], config["version"])
