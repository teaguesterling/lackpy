# write_file → Path.write_text

The `write_file` tool was removed with the tool kits. A program writes a file
with `Path(p).write_text(s)`. The [path policy](../concepts/paths.md) checks
the write first.

```python
import json
from pathlib import Path

report = {"files": 3, "lines": 120}
Path("out").mkdir(exist_ok=True)
Path("out/report.json").write_text(json.dumps(report))
```

Use `Path(p).write_bytes(b)` for binary data. `mkdir(parents=True)` checks
each directory that it makes. A write to a settings location, such as any
`.claude` or `.lackpy` directory, always refuses.
