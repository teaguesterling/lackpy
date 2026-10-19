# /// intent
# goal   = "Replace TODO comment in cpp files and print changed file names"
# reads  = ["src/*.cpp"]
# writes = ["src/*.cpp"]
# output = "printed"
# ///

from pathlib import Path

src = Path('src')
pattern = '*.cpp'
files = src.glob(pattern)
for f in files:
    text = f.read_text()
    old = '// TODO: remove'
    new = '// done'
    if old in text:
        updated = text.replace(old, new)
        f.write_text(updated)
        name = f.name
        print(name)
