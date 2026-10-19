# /// intent
# goal   = "Read the features from config.json and join them with a default separator of ', '"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def join_features(features, sep=', '):
    return sep.join(features)

# Load config from data/config.json
config_path = Path("data/config.json")
config_text = config_path.read_text()
config = json.loads(config_text)

# Extract features and join with default separator
features = config["features"]
result = join_features(features)

# Print the joined features
print(result)
