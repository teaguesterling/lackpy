# /// intent
# goal   = "Print the name and version from data/config.json separated by a space"
# reads  = ["data/config.json"]
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

p = Path('data/config.json')
text = p.read_text()
data = json.loads(text)
name = data['name']
version = data['version']
print(f'{name} {version}')
