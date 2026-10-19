# TODO scan

A helper counts the TODO lines of one file.

```lackpy
from pathlib import Path


def todo_count(path):
    lines = path.read_text().splitlines()
    return sum(1 for line in lines if "TODO" in line)
```

Then the scan runs it over every `.cpp` file under `src`.

```lackpy
totals = {}
for p in sorted(Path("src").glob("*.cpp")):
    totals[p.name] = todo_count(p)
    print(p.name, totals[p.name])
```

The files hold {sum(totals.values())} TODO line(s) in all.
