from pathlib import Path
from collections import Counter

# Find all .cpp files under src
paths = Path("src").glob("**/*.cpp")

# Process each file
for p in sorted(paths):
    text = p.read_text()
    # Check if the TODO line exists
    if "// TODO: remove" in text:
        # Replace the line
        new_text = text.replace("// TODO: remove", "// done")
        # Write back to the file
        p.write_text(new_text)
        # Print the filename
        print(p.name)
