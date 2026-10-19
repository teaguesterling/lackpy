# /// intent
# goal   = "Print the total number of lines across all files under src"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

src_dir = Path('src')
total_lines = 0

for file_path in src_dir.iterdir():
    if file_path.is_file():
        text = file_path.read_text()
        lines = text.split('\n')
        count = len(lines)
        if text.endswith('\n'):
            count = count - 1
        total_lines = total_lines + count

print(total_lines)
