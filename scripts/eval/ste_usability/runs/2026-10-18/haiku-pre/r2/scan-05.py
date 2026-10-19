# /// intent
# goal   = "Write report.txt with .cpp files and character counts"
# reads  = ["src/*.cpp"]
# writes = ["report.txt"]
# output = "none"
# ///

from pathlib import Path


lines = []
paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    char_count = len(text)
    line = f"{p.name}: {char_count}"
    lines.append(line)
report = "\n".join(lines)
Path("report.txt").write_text(report)
