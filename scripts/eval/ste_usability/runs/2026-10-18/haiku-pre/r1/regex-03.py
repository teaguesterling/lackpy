# /// intent
# goal   = "Replace code=XNN with code=[XNN] in logs/app.log and write to logs/app.marked.log"
# reads  = ["logs/app.log"]
# writes = ["logs/app.marked.log"]
# output = "none"
# ///

import re
from pathlib import Path

text = Path("logs/app.log").read_text()
marked = re.sub(r"code=([A-Z]\d+)", r"code=[\1]", text)
Path("logs/app.marked.log").write_text(marked)
