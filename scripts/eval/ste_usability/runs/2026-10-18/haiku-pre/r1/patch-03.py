# /// intent
# goal   = "In every .cpp file under src, replace '// TODO: remove' with '// done' and print each changed file."
# reads  = ["src/*.cpp"]
# writes = ["src/*.cpp"]
# output = "printed"
# ///

from pathlib import Path

paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    if "// TODO: remove" in text:
        new_text = text.replace("// TODO: remove", "// done")
        p.write_text(new_text)
        print(p.name)
