# /// intent
# goal   = "Print the name and version from data/config.json, separated by a space."
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the config file
config_data = json.loads(Path("data/config.json").read_text())

# Extract name and version
name = config_data["name"]
version = config_data["version"]

# Print name and version separated by a space
print(f"{name} {version}")
