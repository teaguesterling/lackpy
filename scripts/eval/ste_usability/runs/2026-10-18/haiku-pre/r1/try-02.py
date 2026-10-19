# /// intent
# goal   = "Print the contents of missing.txt, or the word 'absent' if the file does not exist."
# reads  = []
# writes = []
# output = "printed"
# ///

from pathlib import Path

path = Path("missing.txt")
if path.exists():
    text = path.read_text()
    print(text)
else:
    print("absent")
