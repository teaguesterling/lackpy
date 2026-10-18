from pathlib import Path


def headings(text):
    lines = text.splitlines()
    return [line for line in lines if line.startswith("#")]


path = Path("notes.md")
text = path.read_text()
lines = headings(text)
print("\n".join(lines))
