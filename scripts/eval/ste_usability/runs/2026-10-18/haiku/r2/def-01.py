# /// intent
# goal   = "Count TODO markers in each .cpp file under src directory"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

def count_todos(text):
    return text.count("TODO")

src_dir = Path("src")
cpp_files = src_dir.glob("*.cpp")
cpp_files_sorted = sorted(cpp_files)

for cpp_file in cpp_files_sorted:
    text = cpp_file.read_text()
    count = count_todos(text)
    print(f"{cpp_file.name}: {count}")
