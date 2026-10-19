from pathlib import Path

# /// intent
# goal   = "Count the lines that contain TODO in each .cpp file under src. Print each file name and its count, sorted by file name."
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    count = sum(1 for line in text.splitlines() if "TODO" in line)
    print(p.name, count)
