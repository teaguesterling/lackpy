# /// intent
# goal   = "Join config features from JSON with a default separator"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

def join_strings(strings, separator=', '):
    return separator.join(strings)

text = Path('data/config.json').read_text()
config = json.loads(text)
features = config['features']
result = join_strings(features)
print(result)
