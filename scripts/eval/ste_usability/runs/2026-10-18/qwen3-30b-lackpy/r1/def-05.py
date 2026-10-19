from pathlib import Path
import json

def join_strings(strings, sep=', '):
    return sep.join(strings)

# Read the config features from data/config.json
config_path = Path("data/config.json")
config_text = config_path.read_text()
config_features = json.loads(config_text)

# Join the features with the default separator and print
joined_features = join_strings(config_features)
print(joined_features)
