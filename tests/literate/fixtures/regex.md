# Error codes

Find each distinct error code in the ERROR lines of the log.

```lackpy
import re
from collections import Counter
from pathlib import Path

text = Path("logs/app.log").read_text()
codes = re.findall(r"ERROR code=(E\d+)", text)
counts = Counter(codes)
```

The log has {len(codes)} ERROR line(s) with {len(counts)} distinct code(s).

```lackpy
for code in sorted(counts):
    print(code, counts[code])
```
