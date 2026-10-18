# /// intent
# goal   = "Print the contents of missing.txt, or the word 'absent' if the file does not exist."
# reads  = ["missing.txt"]
# writes = []
# output = "printed"
# ///

import pathlib

p = pathlib.Path("missing.txt")
if p.exists():
    text = p.read_text()
    print(text)
else:
    print("absent")
