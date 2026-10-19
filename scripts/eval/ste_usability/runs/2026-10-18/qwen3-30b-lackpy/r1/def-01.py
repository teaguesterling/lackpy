from pathlib import Path
import json

def todo_count(text):
    return text.count("TODO")

paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    count = todo_count(text)
    print(p.name, count)
