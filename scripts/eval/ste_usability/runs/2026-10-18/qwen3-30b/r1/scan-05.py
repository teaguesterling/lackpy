# /// intent
# goal   = "Generate a report with one line per .cpp file under src, sorted by name, showing the filename and its character count."
# reads  = ["src/a.cpp", "src/b.cpp", "src/c.cpp"]
# writes = ["report.txt"]
# output = "printed"
# ///

import json
import re
from pathlib import Path

# Get all .cpp files in src directory
cpp_files = sorted(Path("src").glob("*.cpp"))

# Process each file: read its content and count characters
report_lines = []
for file_path in cpp_files:
    content = file_path.read_text()
    char_count = len(content)
    report_lines.append(f"{file_path.name}: {char_count}")

# Write the report to report.txt
Path("report.txt").write_text("\n".join(report_lines))
