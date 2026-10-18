# /// intent
# goal   = "Print the name and version from data/config.json, separated by a space"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the config file
config_path = Path("data/config.json")
config_text = config_path.read_text()

# Parse the JSON
config = json.loads(config_text)

# Extract name and version
name = config["name"]
version = config["version"]

# Print the result
print(f"{name} {version}")
