# /// intent
# goal   = "Print the names of the functions defined in the .cpp files under src, sorted."
# reads  = ["src/a.cpp", "src/b.cpp", "src/c.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path
import re

def extract_names(dir_path):
    names = []
    path_obj = Path(dir_path)
    if not path_obj.is_dir():
        return names
    for file_path in path_obj.iterdir():
        if file_path.suffix == '.cpp':
            text = file_path.read_text()
            matches = re.findall(r'^\s*int\s+(\w+)\s*\(', text, re.MULTILINE)
            names.extend(matches)
    return names

src_names = extract_names('src')
src_names.sort()
for name in src_names:
    print(name)
