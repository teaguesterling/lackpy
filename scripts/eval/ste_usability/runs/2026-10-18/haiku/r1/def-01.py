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
cpp_files = sorted(src_dir.glob("*.cpp"))

for cpp_file in cpp_files:
    text = cpp_file.read_text()
    count = count_todos(text)
    print(f"{cpp_file.name}: {count}")
