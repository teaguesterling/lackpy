from pathlib import Path


def headings(text):
    return [line for line in text.splitlines() if line.startswith("#")]


path = Path("notes.md")
text = path.read_text()
print("\n".join(headings(text)))
