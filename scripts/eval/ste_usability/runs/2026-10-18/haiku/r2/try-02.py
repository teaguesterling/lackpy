# /// intent
# goal   = "Print the contents of missing.txt, or 'absent' if the file does not exist"
# reads  = ["missing.txt"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

p = Path("missing.txt")
if p.exists():
    text = p.read_text()
    print(text)
else:
    print("absent")
