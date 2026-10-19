# /// intent
# goal   = "Add the line '- release notes' after the line '- changelog' in notes.md, failing if '- changelog' does not occur exactly once"
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

import json
from pathlib import Path

# Read the file content
path = Path("notes.md")
content = path.read_text()

# Split the content into lines
lines = content.splitlines()

# Find the index of '- changelog'
changelog_index = -1
for i, line in enumerate(lines):
    if line.strip() == "- changelog":
        if changelog_index != -1:
            # Found more than one occurrence
            raise RuntimeError("'- changelog' does not occur exactly once")
        changelog_index = i

# If not found, raise an error
if changelog_index == -1:
    raise RuntimeError("'- changelog' does not occur exactly once")

# Insert the new line after '- changelog'
lines.insert(changelog_index + 1, "- release notes")

# Join back into a single string
new_content = "\n".join(lines)

# Write back to the file
path.write_text(new_content)
