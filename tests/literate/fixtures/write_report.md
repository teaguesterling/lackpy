# Size report

Collect the size of every `.cpp` file under `src`.

```lackpy
from pathlib import Path

lines = []
for p in sorted(Path("src").glob("*.cpp")):
    lines.append(f"{p.name}: {len(p.read_text())}")
```

The report has {len(lines)} line(s). A note goes next to it, written with an
annotation.

```lackpy @write(out/README.txt)
The sizes of the .cpp files under src, in characters.
```

```lackpy
Path("out/report.txt").write_text("\n".join(lines) + "\n")
print(Path("out/report.txt").read_text(), end="")
```
