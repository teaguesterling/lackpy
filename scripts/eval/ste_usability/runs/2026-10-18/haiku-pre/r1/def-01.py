# /// intent
# goal   = "Count TODO markers in each .cpp file under src and print sorted by name"
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
    count = todo_count(text)
    print(p.name, count)
