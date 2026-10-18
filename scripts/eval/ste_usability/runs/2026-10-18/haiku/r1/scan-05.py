# /// intent
# goal   = "Write report.txt with one line per .cpp file under src with character count"
# reads  = ["src/*.cpp"]
# writes = ["report.txt"]
# output = "none"
# ///

from pathlib import Path

src_dir = Path('src')
report_lines = []

for cpp_file in sorted(src_dir.glob('*.cpp')):
    text = cpp_file.read_text()
    char_count = len(text)
    report_lines.append(f"{cpp_file.name}: {char_count}")

report_text = '\n'.join(report_lines)
report_path = Path('report.txt')
report_path.write_text(report_text)
