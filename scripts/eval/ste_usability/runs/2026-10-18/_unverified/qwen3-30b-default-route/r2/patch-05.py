# /// intent
# goal   = "Add the line '- release notes' after the line '- changelog' in notes.md, failing if '- changelog' does not occur exactly once"
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///
import json
import re
from pathlib import Path

# Read the file content
content = Path("notes.md").read_text()

# Split the content into lines
lines = content.splitlines()

# Find the index of '- changelog'
changelog_index = -1
for i, line in enumerate(lines):
    if line.strip() == "- changelog":
        if changelog_index != -1:
            # Found more than one '- changelog'
            print("Expected exactly one '- changelog' line", file=sys.stderr)
            exit(1)
        changelog_index = i

# If not found, raise an error
if changelog_index == -1:
    print("Expected exactly one '- changelog' line", file=sys.stderr)
    exit(1)

# Insert the new line after '- changelog'
lines.insert(changelog_index + 1, "- release notes")

# Join back into a single string
new_content = "\n".join(lines)

# Write back to the file
Path("notes.md").write_text(new_content)
