# /// intent
# goal   = "Print the names of the files under src (any extension) that contain no TODO, sorted."
# reads  = ["pathlib.Path.glob", "pathlib.Path.read_text"]
# writes = ["print"]
# output = "printed"
# ///

import pathlib

def check_files(root):
    files = list(root.glob('*.c*'))
    results = []
    for p in files:
        text = p.read_text()
        if 'TODO' not in text:
            results.append(p.name)
    results.sort()
    return results

if __name__ == '__main__':
    results = check_files(pathlib.Path('src'))
    print(*results)
