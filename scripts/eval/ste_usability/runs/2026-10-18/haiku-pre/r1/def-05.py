# /// intent
# goal   = "Join the features from data/config.json with a default separator"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path


def join_strings(items, sep=", "):
    return sep.join(items)


text = Path("data/config.json").read_text()
data = json.loads(text)
features = data["features"]
result = join_strings(features)
print(result)
