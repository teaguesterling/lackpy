from pathlib import Path
import json

total_lines = 0
for p in Path("src").rglob("*"):
    if p.is_file():
        try:
            lines = p.read_text().splitlines()
            total_lines += len(lines)
        except (OSError, UnicodeDecodeError):
            continue
print(total_lines)
