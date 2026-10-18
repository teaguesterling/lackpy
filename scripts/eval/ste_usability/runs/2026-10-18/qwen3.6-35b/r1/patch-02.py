# /// intent
# goal   = "Change 'Status: draft' to 'Status: final' in notes.md, failing if the line does not occur exactly once."
# reads  = ["notes.md"]
# writes = ["notes.md"]
# output = "none"
# ///

import json

def main():
    path_str = "notes.md"
    text = open_work(path_str)
    lines = text.split("\n")
    new_lines = []
    found = 0
    for line in lines:
        if line == "Status: draft":
            found += 1
            if found > 1:
                raise SystemExit("Status line occurs more than once")
            new_lines.append("Status: final")
        else:
            new_lines.append(line)
    if found != 1:
        raise SystemExit("Status line does not occur exactly once")
    new_text = "\n".join(new_lines)
    write_work(path_str, new_text)

def open_work(path_str):
    from pathlib import Path
    p = Path(path_str)
    return p.read_text()

def write_work(path_str, content):
    from pathlib import Path
    p = Path(path_str)
    p.write_text(content)

main()
