# /// intent
# goal   = "Count lines containing TODO in each .cpp file under src, print sorted by filename"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

src_dir = Path('src')
counts = {}

for cpp_file in sorted(src_dir.glob('*.cpp')):
    text = cpp_file.read_text()
    lines = text.split('\n')
    count = sum(1 for line in lines if 'TODO' in line)
    counts[cpp_file.name] = count

for name in sorted(counts):
    print(f"{name} {counts[name]}")
