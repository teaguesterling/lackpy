# /// intent
# goal   = "Print names of files under src with no TODO, sorted"
# reads  = ["src/*"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

src_dir = Path('src')
no_todo_files = []

for file_path in src_dir.iterdir():
    if file_path.is_file():
        text = file_path.read_text()
        if 'TODO' not in text:
            no_todo_files.append(file_path.name)

for name in sorted(no_todo_files):
    print(name)
