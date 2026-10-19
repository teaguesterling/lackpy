from pathlib import Path
import json

# /// intent
# goal   = "Replace a specific docx note in registry.cpp with a revised note, ensuring the old note occurs exactly once"
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

path = Path("registry.cpp")
text = path.read_text()
assert text.count("images and footnotes are not read yet") == 1, "the old note must occur exactly once"
path.write_text(text.replace("images and footnotes are not read yet", "footnotes are not read yet"))
