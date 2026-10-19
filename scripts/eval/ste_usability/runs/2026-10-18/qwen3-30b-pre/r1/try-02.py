# /// intent
# goal   = "Print the contents of missing.txt, or the word 'absent' if the file does not exist"
# reads  = ["missing.txt"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

p = Path("missing.txt")
if p.exists():
    print(p.read_text())
else:
    print("absent")
