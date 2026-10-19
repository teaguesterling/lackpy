"""Synthetic tasks for the STE usability eval.

Each task is one intent in the six corpus families (design section 2.2). The
`reference` is a lackpy program written by Claude. It must validate, and its
behaviour on a fresh copy of fixtures/ defines the expected result. The tasks
are synthetic: no content comes from session history.
"""

TASKS = [
    # --- guarded patching -------------------------------------------------
    dict(id="patch-01", family="patch",
         goal="In registry.cpp, replace the docx note 'images and footnotes are not read yet' with 'footnotes are not read yet'. Fail if the old note does not occur exactly once.",
         reads=["registry.cpp"], writes=["registry.cpp"],
         reference='''from pathlib import Path

path = Path("registry.cpp")
text = path.read_text()
old = "images and footnotes are not read yet"
assert text.count(old) == 1, "the docx note occurs exactly once"
text = text.replace(old, "footnotes are not read yet")
path.write_text(text)
'''),
    dict(id="patch-02", family="patch",
         goal="In notes.md, change the line 'Status: draft' to 'Status: final'. Fail if the line does not occur exactly once.",
         reads=["notes.md"], writes=["notes.md"],
         reference='''from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
'''),
    dict(id="patch-03", family="patch",
         goal="In every .cpp file under src, replace '// TODO: remove' with '// done'. Print the name of each file that changed.",
         reads=["src/*.cpp"], writes=["src/*.cpp"],
         reference='''from pathlib import Path

paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    if "// TODO: remove" in text:
        p.write_text(text.replace("// TODO: remove", "// done"))
        print(p.name)
'''),
    dict(id="patch-04", family="patch",
         goal="In data/config.json, set the version to 1.3.0 and write the file back as JSON with an indent of 2.",
         reads=["data/config.json"], writes=["data/config.json"],
         reference='''import json
from pathlib import Path

path = Path("data/config.json")
text = path.read_text()
config = json.loads(text)
config["version"] = "1.3.0"
path.write_text(json.dumps(config, indent=2))
'''),
    dict(id="patch-05", family="patch",
         goal="In notes.md, add the line '- release notes' after the line '- changelog'. Fail if '- changelog' does not occur exactly once.",
         reads=["notes.md"], writes=["notes.md"],
         reference='''from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("- changelog") == 1, "the changelog item occurs once"
path.write_text(text.replace("- changelog", "- changelog\\n- release notes"))
'''),
    # --- structured reads -------------------------------------------------
    dict(id="json-01", family="json",
         goal="Print the name and version from data/config.json, separated by a space.",
         reads=["data/config.json"], writes=[],
         reference='''import json
from pathlib import Path

text = Path("data/config.json").read_text()
config = json.loads(text)
print(config["name"], config["version"])
'''),
    dict(id="json-02", family="json",
         goal="Count the rows in data/rows.jsonl by status, skipping lines that are not valid JSON. Print each status and its count, sorted by status.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from collections import Counter
from pathlib import Path

counts = Counter()
for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    counts[row["status"]] += 1
for status in sorted(counts):
    print(status, counts[status])
'''),
    dict(id="json-03", family="json",
         goal="Print the sum of the size field over all valid rows in data/rows.jsonl. Skip lines that are not valid JSON.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path

total = 0
for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    total += row["size"]
print(total)
'''),
    dict(id="json-04", family="json",
         goal="Print the ids of rows in data/rows.jsonl whose status is fail, one per line, in file order. Skip invalid lines.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path

for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row["status"] == "fail":
        print(row["id"])
'''),
    dict(id="json-05", family="json",
         goal="Write data/ok.json: a JSON list of the ids of rows in data/rows.jsonl with status ok. Skip invalid lines.",
         reads=["data/rows.jsonl"], writes=["data/ok.json"],
         reference='''import json
from pathlib import Path

ids = []
for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row["status"] == "ok":
        ids.append(row["id"])
Path("data/ok.json").write_text(json.dumps(ids))
'''),
    # --- scans --------------------------------------------------------------
    dict(id="scan-01", family="scan",
         goal="Count the lines that contain TODO in each .cpp file under src. Print each file name and its count, sorted by file name.",
         reads=["src/*.cpp"], writes=[],
         reference='''from pathlib import Path

paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    lines = p.read_text().splitlines()
    n = sum(1 for line in lines if "TODO" in line)
    print(p.name, n)
'''),
    dict(id="scan-02", family="scan",
         goal="Print the names of the files under src (any extension) that contain no TODO, sorted.",
         reads=["src/*"], writes=[],
         reference='''from pathlib import Path

paths = Path("src").iterdir()
for p in sorted(paths):
    text = p.read_text()
    if "TODO" not in text:
        print(p.name)
'''),
    dict(id="scan-03", family="scan",
         goal="Print the total number of lines across all files under src.",
         reads=["src/*"], writes=[],
         reference='''from pathlib import Path

total = 0
for p in Path("src").iterdir():
    text = p.read_text()
    total += len(text.splitlines())
print(total)
'''),
    dict(id="scan-04", family="scan",
         goal="Print every file suffix used under src with the number of files that have it, sorted by suffix.",
         reads=["src/*"], writes=[],
         reference='''from collections import Counter
from pathlib import Path

counts = Counter(p.suffix for p in Path("src").iterdir())
for suffix in sorted(counts):
    print(suffix, counts[suffix])
'''),
    dict(id="scan-05", family="scan",
         goal="Write report.txt with one line per .cpp file under src, sorted by name, in the form '<name>: <number of characters>'.",
         reads=["src/*.cpp"], writes=["report.txt"],
         reference='''from pathlib import Path

lines = []
paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    lines.append(f"{p.name}: {len(text)}")
Path("report.txt").write_text("\\n".join(lines) + "\\n")
'''),
    # --- regex --------------------------------------------------------------
    dict(id="regex-01", family="regex",
         goal="Print each distinct error code (like E12) from the ERROR lines of logs/app.log with how often it occurs, sorted by code.",
         reads=["logs/app.log"], writes=[],
         reference='''import re
from collections import Counter
from pathlib import Path

text = Path("logs/app.log").read_text()
codes = re.findall(r"ERROR code=(E\\d+)", text)
counts = Counter(codes)
for code in sorted(counts):
    print(code, counts[code])
'''),
    dict(id="regex-02", family="regex",
         goal="Print the time (HH:MM:SS) of the first and the last line of logs/app.log, separated by a space.",
         reads=["logs/app.log"], writes=[],
         reference='''import re
from pathlib import Path

lines = Path("logs/app.log").read_text().splitlines()
first = re.search(r"\\d\\d:\\d\\d:\\d\\d", lines[0]).group(0)
last = re.search(r"\\d\\d:\\d\\d:\\d\\d", lines[-1]).group(0)
print(first, last)
'''),
    dict(id="regex-03", family="regex",
         goal="In logs/app.log, replace every 'code=XNN' with 'code=[XNN]' (X is a letter, NN digits) and write the result to logs/app.marked.log.",
         reads=["logs/app.log"], writes=["logs/app.marked.log"],
         reference='''import re
from pathlib import Path

text = Path("logs/app.log").read_text()
marked = re.sub(r"code=([A-Z]\\d+)", r"code=[\\1]", text)
Path("logs/app.marked.log").write_text(marked)
'''),
    dict(id="regex-04", family="regex",
         goal="Print the names of the functions defined in the .cpp files under src (the word before '(' on lines that start with 'int '), sorted.",
         reads=["src/*.cpp"], writes=[],
         reference='''import re
from pathlib import Path

names = []
for p in Path("src").glob("*.cpp"):
    text = p.read_text()
    names.extend(re.findall(r"^int (\\w+)\\(", text, re.M))
for name in sorted(names):
    print(name)
'''),
    dict(id="regex-05", family="regex",
         goal="Print the number of lines in logs/app.log whose level is WARN or ERROR.",
         reads=["logs/app.log"], writes=[],
         reference='''import re
from pathlib import Path

lines = Path("logs/app.log").read_text().splitlines()
n = sum(1 for line in lines if re.search(r" (WARN|ERROR) ", line))
print(n)
'''),
    # --- try ----------------------------------------------------------------
    dict(id="try-01", family="try",
         goal="Print the value of the 'owner' key in data/config.json, or 'none' if the key is missing.",
         reads=["data/config.json"], writes=[],
         reference='''import json
from pathlib import Path

text = Path("data/config.json").read_text()
config = json.loads(text)
try:
    owner = config["owner"]
except KeyError:
    owner = "none"
print(owner)
'''),
    dict(id="try-02", family="try",
         goal="Print the contents of missing.txt, or the word 'absent' if the file does not exist.",
         reads=["missing.txt"], writes=[],
         reference='''from pathlib import Path

try:
    text = Path("missing.txt").read_text()
except FileNotFoundError:
    text = "absent"
print(text)
'''),
    dict(id="try-03", family="try",
         goal="For each line of data/rows.jsonl, print 'bad' if the line is not valid JSON, else the row id.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path

for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print("bad")
        continue
    print(row["id"])
'''),
    dict(id="try-04", family="try",
         goal="Convert each word of the text 'a 3 b 7 9' to an integer where possible, skip the others, and print the sum.",
         reads=[], writes=[],
         reference='''total = 0
for word in "a 3 b 7 9".split():
    try:
        n = int(word)
    except ValueError:
        continue
    total += n
print(total)
'''),
    dict(id="try-05", family="try",
         goal="Print the first tag of each valid row in data/rows.jsonl, or '-' when the row has no tags. Skip invalid lines.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path

for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    try:
        tag = row["tags"][0]
    except IndexError:
        tag = "-"
    print(tag)
'''),
    # --- def ----------------------------------------------------------------
    dict(id="def-01", family="def",
         goal="Define a function that returns the number of TODO markers in a text, and use it to print the count for each .cpp file under src, sorted by name.",
         reads=["src/*.cpp"], writes=[],
         reference='''from pathlib import Path


def todo_count(text):
    return text.count("TODO")


paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    print(p.name, todo_count(text))
'''),
    dict(id="def-02", family="def",
         goal="Define a function that loads the valid rows of a JSONL text, skipping bad lines, and use it to print how many valid rows data/rows.jsonl has.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path


def load_rows(text):
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


text = Path("data/rows.jsonl").read_text()
rows = load_rows(text)
print(len(rows))
'''),
    dict(id="def-03", family="def",
         goal="Define a function that classifies a size as small (under 100), medium (under 200) or large. Print the class of each valid row's size in data/rows.jsonl, in file order.",
         reads=["data/rows.jsonl"], writes=[],
         reference='''import json
from pathlib import Path


def size_class(size):
    classes = {True: "small", False: "medium"}
    if size >= 200:
        return "large"
    return classes[size < 100]


for line in Path("data/rows.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    print(size_class(row["size"]))
'''),
    dict(id="def-04", family="def",
         goal="Define a function that returns the heading lines (lines starting with '#') of a markdown text. Print the headings of notes.md.",
         reads=["notes.md"], writes=[],
         reference='''from pathlib import Path


def headings(text):
    return [line for line in text.splitlines() if line.startswith("#")]


text = Path("notes.md").read_text()
for h in headings(text):
    print(h)
'''),
    dict(id="def-05", family="def",
         goal="Define a function with a default separator of ', ' that joins a list of strings. Print the config features from data/config.json joined with it.",
         reads=["data/config.json"], writes=[],
         reference='''import json
from pathlib import Path


def join_all(items, sep=", "):
    return sep.join(items)


text = Path("data/config.json").read_text()
config = json.loads(text)
print(join_all(config["features"]))
'''),
]
