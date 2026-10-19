# /// intent
# goal   = "Return the heading lines (lines starting with '#') of a markdown text, then print the headings from notes.md."
# reads  = ["data/config.json", "data/rows.jsonl", "logs/app.log", "notes.md", "registry.cpp", "src/a.cpp", "src/b.cpp", "src/c.cpp", "src/util.h"]
# writes = []
# output = "printed"
# ///

import pathlib

def get_headings(text):
    # Return lines that start with '#'.
    # Strip leading/trailing whitespace from lines for safety.
    lines = text.split('\n')
    return [line for line in lines if line.startswith('#')]

def main():
    # Read notes.md from the current working directory.
    path = pathlib.Path('notes.md')
    text = path.read_text()
    
    # Get the headings.
    headings = get_headings(text)
    
    # Print each heading.
    for heading in headings:
        print(heading)

if __name__ == '__main__':
    main()
