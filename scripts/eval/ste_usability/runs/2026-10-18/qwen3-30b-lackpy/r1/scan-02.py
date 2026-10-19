from pathlib import Path

# /// intent
# goal   = "Print the names of files under src (any extension) that contain no TODO, sorted."
# reads  = ["src/"]
# writes = []
# output = "printed"
# ///

paths = Path("src").glob("**/*")
files_without_todo = []

for p in sorted(paths):
    if p.is_file():
        try:
            text = p.read_text()
            if "TODO" not in text.upper():
                files_without_todo.append(p.name)
        except (UnicodeDecodeError, IsADirectoryError):
            continue

for name in sorted(files_without_todo):
    print(name)
