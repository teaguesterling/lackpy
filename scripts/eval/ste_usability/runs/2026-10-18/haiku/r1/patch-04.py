# /// intent
# goal   = "Update version in config.json to 1.3.0 and write with indent of 2"
# reads  = ["data/config.json"]
# writes = ["data/config.json"]
# output = "none"
# ///

import json
from pathlib import Path

p = Path('data/config.json')
text = p.read_text()
d = json.loads(text)
d['version'] = '1.3.0'
updated = json.dumps(d, indent=2)
p.write_text(updated)
