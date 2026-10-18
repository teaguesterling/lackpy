# /// intent
# goal   = "Write report.txt with one line per .cpp file under src with character count"
# reads  = ["src/*.cpp"]
# writes = ["report.txt"]
# output = "none"
# ///

from pathlib import Path

src_dir = Path('src')
cpp_files = src_dir.glob('*.cpp')
cpp_files_sorted = sorted(cpp_files)
report_lines = []

for cpp_file in cpp_files_sorted:
    text = cpp_file.read_text()
    char_count = len(text)
    report_lines.append(f"{cpp_file.name}: {char_count}")

report_text = '\n'.join(report_lines)
report_path = Path('report.txt')
report_path.write_text(report_text)
