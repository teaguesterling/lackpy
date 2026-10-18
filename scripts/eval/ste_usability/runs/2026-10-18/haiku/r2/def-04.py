# /// intent
# goal   = "Extract heading lines from a markdown file"
# reads  = ["notes.md"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

def get_headings(text):
    headings = []
    for line in text.split('\n'):
        if line.startswith('#'):
            headings.append(line)
    return headings

text = Path('notes.md').read_text()
headings = get_headings(text)
for heading in headings:
    print(heading)
