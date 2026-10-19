# /// intent
# goal   = "Replace the note about docx files with a new note, ensuring the old note occurs exactly once"
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

import re
from pathlib import Path

# Read the registry.cpp file
content = Path("registry.cpp").read_text()

# Define the old and new notes
old_note = "images and footnotes are not read yet"
new_note = "footnotes are not read yet"

# Find all occurrences of the old note
matches = re.findall(re.escape(old_note), content)

# Check if the old note occurs exactly once
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one occurrence of '{old_note}', found {len(matches)}")

# Replace the old note with the new one
updated_content = content.replace(old_note, new_note)

# Write the updated content back to registry.cpp
Path("registry.cpp").write_text(updated_content)
