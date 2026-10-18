# /// intent
# goal   = "Count TODO lines in each .cpp file under src and print name with count"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path


def todo_count(text):
    return text.count("TODO")


paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    print(p.name, todo_count(text))
