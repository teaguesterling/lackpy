# /// intent
# goal   = "Update the version in data/config.json to 1.3.0 and write it back with JSON formatting using an indent of 2"
# reads  = ["data/config/config.json"]
# writes = ["data/config.json"]
# output = "none"
# ///

import json
from pathlib import Path

# Read the config file
config_path = Path("data/config.json")
text = config_path.read_text()
config_data = json.loads(text)

# Update the version
config_data["version"] = "1.3.0"

# Write back with indent=2
output = json.dumps(config_data, indent=2)
config_path.write_text(output)
