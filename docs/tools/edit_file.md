# edit_file → read, check, write

The `edit_file` tool was removed with the tool kits. A program edits a file in
three steps: read it, assert that the change applies once, then write it.

```python
from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
```

If the `assert` fails, the program stops before it writes. The
[tutorial](../tutorial.md#2-patch-one-occurrence-in-a-file) runs this example.
