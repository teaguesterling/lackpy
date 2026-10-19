# /// intent
# goal   = "Replace the note for docx from 'images and footnotes are not read yet' to 'footnotes are not read yet' in registry.cpp, failing if the old note does not occur exactly once"
# reads  = ["registry.cpp"]
# writes = ["registry.cpp"]
# output = "none"
# ///

import re
from pathlib import Path

# Read the file
content = Path("registry.cpp").read_text()

# Define the old and new notes
old_note = "images and footnotes are not read yet"
new_note = "footnotes are not read yet"

# Count occurrences of the old note
count = content.count(old_note)

# Check if it occurs exactly once
if count != 1:
    # Use print to report the error since raise is forbidden
    print(f"Expected exactly one occurrence of '{old_note}', found {count}", file=sys.stderr)
    exit(1)

# Replace the note
updated_content = content.replace(old_note, new_note)

# Write back to the file
Path("registry.cpp").write_text(updated_content)
