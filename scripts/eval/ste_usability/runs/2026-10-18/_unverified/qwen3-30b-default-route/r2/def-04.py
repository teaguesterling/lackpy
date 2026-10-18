# /// intent
# goal   = "Extract and print the heading lines (starting with '#') from the markdown file notes.md"
# reads  = ["notes.md"]
# writes = []
# output = "printed"
# ///
import re
from pathlib import Path

def extract_headings(text: str) -> list[str]:
    """Extract lines that start with '#' from the given text."""
    return [line for line in text.splitlines() if line.strip().startswith('#')]

# Read the markdown file
content = Path('notes.md').read_text()

# Extract and print headings
headings = extract_headings(content)
for heading in headings:
    print(heading)
