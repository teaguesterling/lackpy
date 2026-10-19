from pathlib import Path

# /// intent
# goal   = "Write report.txt with one line per .cpp file under src, sorted by name, in the form '<name>: <number of characters>'"
# reads  = ["src/*.cpp"]
# writes = ["report.txt"]
# output = "printed"
# ///

paths = Path("src").glob("*.cpp")
lines = []
for p in sorted(paths):
    text = p.read_text()
    lines.append(f"{p.name}: {len(text)}")

Path("report.txt").write_text("\n".join(lines))
