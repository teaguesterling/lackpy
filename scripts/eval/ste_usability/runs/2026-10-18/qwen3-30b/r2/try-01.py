# /// intent
# goal   = "Print the value of the 'owner' key in data/config.json, or 'none' if the key is missing"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the config file
config_path = Path("data/config.json")
text = config_path.read_text()
config_data = json.loads(text)

# Extract the owner value or default to 'none'
owner = config_data.get("owner", "none")

# Print the result
print(owner)
