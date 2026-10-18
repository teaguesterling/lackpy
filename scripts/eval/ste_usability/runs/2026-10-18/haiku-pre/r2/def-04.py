# /// intent
# goal   = "Extract and print all heading lines from notes.md"
# reads  = ["notes.md"]
# writes = []
# output = "printed"
# ///

from pathlib import Path


def heading_lines(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(line)
    return lines


text = Path("notes.md").read_text()
headings = heading_lines(text)
for heading in headings:
    print(heading)
