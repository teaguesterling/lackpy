# /// intent
# goal   = "Print the names of the files under src (any extension) that contain no TODO, sorted."
# reads  = ["pathlib.Path.glob", "pathlib.Path.read_text"]
# writes = ["print"]
# output = "printed"
# ///

import pathlib

root = pathlib.Path('src')
candidates = list(root.glob('*.c*'))
results = []
for p in candidates:
    text = p.read_text()
    if 'TODO' not in text:
        results.append(p.name)
results.sort()
print(*results)
