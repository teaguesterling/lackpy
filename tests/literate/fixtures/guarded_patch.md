# Finalize the notes

The notes file still marks itself as a draft. The patch is guarded: it runs
only when the status line occurs exactly once.

```lackpy
from pathlib import Path

path = Path("notes.md")
text = path.read_text()
count = text.count("Status: draft")
print("draft lines:", count)
```

The draft status line occurs {count} time(s) in {path.name}.

```lackpy
assert count == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
print("patched", path.name)
```

Done: {path.name} now says final.
