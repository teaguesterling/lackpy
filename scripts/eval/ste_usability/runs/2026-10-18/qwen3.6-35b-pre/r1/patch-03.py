# /// intent
# goal   = "Replace 'TODO: remove' with 'done' in .cpp files under src and print changed filenames."
# reads  = ["src/**/*.cpp"]
# writes = ["src/a.cpp", "src/b.cpp"]
# output = "printed"
# ///

from pathlib import Path

def process_file(p):
    text = p.read_text()
    if 'TODO: remove' in text:
        new_text = text.replace('TODO: remove', 'done')
        p.write_text(new_text)
        return True
    return False

for path in sorted(Path("src").glob("*.cpp")):
    if process_file(path):
        print(path.name)
