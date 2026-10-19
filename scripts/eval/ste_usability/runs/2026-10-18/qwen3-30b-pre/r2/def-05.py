# /// intent
# goal   = "Read the features from config.json and join them with a default separator, then print the result."
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
import json
from collections import Counter

# Define the join_features function with a default separator
def join_features(items, sep=', '):
    return sep.join(items)

# Read the config file
config_path = Path("data/config.json")
config_text = config_path.read_text()
config_data = json.loads(config_text)

# Extract the features list
features = config_data["features"]

# Join the features with the default separator and print
result = join_features(features)
print(result)
