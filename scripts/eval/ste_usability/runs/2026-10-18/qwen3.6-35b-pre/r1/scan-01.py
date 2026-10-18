# /// intent
# goal   = "Count the lines that contain TODO in each .cpp file under src. Print each file name and its count, sorted by file name."
# reads  = ["src"]
# writes = []
# output = "printed"
# ///
from pathlib import Path


def count_todos(text):
    return text.count("TODO")


paths = sorted(Path("src").glob("*.cpp"))
for p in paths:
    text = p.read_text()
    print(p.name, count_todos(text))
