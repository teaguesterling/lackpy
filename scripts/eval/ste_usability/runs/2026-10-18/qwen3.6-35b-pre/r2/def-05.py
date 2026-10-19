# /// intent
# goal   = "Define a function that joins strings with a default separator and use it to join config features."
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///
import json
from pathlib import Path

def join_strings(items, sep=', '):
    return sep.join(items)

text = Path("data/config.json").read_text()
config = json.loads(text)
features = config["features"]
result = join_strings(features)
print(result)
