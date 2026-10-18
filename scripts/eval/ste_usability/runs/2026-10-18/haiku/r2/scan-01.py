# /// intent
# goal   = "Count lines containing TODO in each .cpp file under src, print sorted by filename"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

src_dir = Path('src')
cpp_files = src_dir.glob('*.cpp')
cpp_files_sorted = sorted(cpp_files)

for cpp_file in cpp_files_sorted:
    text = cpp_file.read_text()
    lines = text.split('\n')
    count = sum(1 for line in lines if 'TODO' in line)
    print(f"{cpp_file.name} {count}")
