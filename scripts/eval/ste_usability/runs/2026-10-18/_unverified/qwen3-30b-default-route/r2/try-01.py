# /// intent
# goal   = "Print the value of the 'owner' key in data/config.json, or 'none' if the key is missing."
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///
import json
from pathlib import Path

# Load the config file
config_path = Path("data/config.json")
try:
    config_text = config_path.read_text()
    config_data = json.loads(config_text)
except (json.JSONDecodeError, FileNotFoundError):
    # If the file is not valid JSON or doesn't exist, treat as missing owner
    owner = "none"
else:
    # Extract owner or default to 'none' if missing
    owner = config_data.get("owner", "none")

print(owner)
