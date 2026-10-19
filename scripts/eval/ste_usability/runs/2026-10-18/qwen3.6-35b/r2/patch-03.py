# /// intent
# goal   = "In every .cpp file under src, replace '// TODO: remove' with '// done'. Print the name of each file that changed."
# reads  = ["src/*.cpp"]
# writes = ["src/b.cpp"]
# output = "printed"
# ///

import json
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    rows = []
    text = Path(path).read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows

def parse_log(path):
    log_lines = Path(path).read_text().splitlines()
    errors = []
    for line in log_lines:
        parts = line.split(None, 3)
        if len(parts) >= 4:
            if parts[2] == "ERROR":
                code = parts[3].split()[0].split("=")[1] if "=" in parts[3] else parts[3]
                errors.append(code)
    return errors

def main():
    src_dir = Path("src")
    changed_files = []

    if src_dir.is_dir():
        matches = list(src_dir.glob("*.cpp"))
        for cpp_file in sorted(matches):
            content = cpp_file.read_text()
            new_content = content.replace('// TODO: remove', '// done')
            if new_content != content:
                cpp_file.write_text(new_content)
                changed_files.append(cpp_file.name)

    for fname in changed_files:
        print(fname)

main()
