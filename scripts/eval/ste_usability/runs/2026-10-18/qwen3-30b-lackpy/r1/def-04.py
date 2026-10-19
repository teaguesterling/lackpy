from pathlib import Path

def heading_lines(text):
    return [line for line in text.splitlines() if line.strip().startswith('#')]

text = Path("notes.md").read_text()
headings = heading_lines(text)
for line in headings:
    print(line)
