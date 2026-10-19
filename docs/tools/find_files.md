# find_files → Path.glob

The `find_files` tool was removed with the tool kits. A program lists files
with `Path(d).glob(pattern)` or `Path(d).rglob(pattern)`. The
[path policy](../concepts/paths.md) checks each entry.

```python
from pathlib import Path

paths = Path("src").rglob("*.py")
for p in sorted(paths):
    print(p)
```

A listing omits each entry that the path policy refuses for read. A pattern
that is absolute or that contains `..` refuses. `iterdir()` lists one
directory.
