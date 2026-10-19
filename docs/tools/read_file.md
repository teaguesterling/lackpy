# read_file → Path.read_text

The `read_file` tool was removed with the tool kits. A program reads a file
with `Path(p).read_text()`. The [path policy](../concepts/paths.md) checks the
read first.

```python
from pathlib import Path

text = Path("README.md").read_text()
print(len(text.splitlines()))
```

Use `Path(p).read_bytes()` for binary data. A relative path is relative to the
workspace. A missing file raises `FileNotFoundError`, as in plain Python.
