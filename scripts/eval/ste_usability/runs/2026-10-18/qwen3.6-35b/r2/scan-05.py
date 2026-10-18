# /// intent
# goal   = "Write a report.txt with one line per .cpp file under src, sorted by name, in the form '<name>: <number of characters>'."
# reads  = ["src/.cpp"]
# writes = ["report.txt"]
# output = "value"
# ///

from pathlib import Path


def count_chars(p):
    text = p.read_text()
    return len(text)


def main():
    src_dir = Path('src')
    cpp_files = [p for p in src_dir.glob('*.cpp') if p.is_file()]
    cpp_files.sort(key=lambda p: p.name)
    lines = []
    for p in cpp_files:
        n = count_chars(p)
        lines.append(f"{p.name}: {n}")
    text = "\n".join(lines)
    Path('report.txt').write_text(text)


main()
