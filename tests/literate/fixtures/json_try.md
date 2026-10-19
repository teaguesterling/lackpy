# Rows by status

Count the rows of `data/rows.jsonl` by status. A line that is not valid JSON
is skipped.

```lackpy
import json
from collections import Counter
from pathlib import Path

counts = Counter()
lines = Path("data/rows.jsonl").read_text().splitlines()
for line in lines:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    counts[row["status"]] += 1
```

There are {sum(counts.values())} valid rows; {len(lines) - sum(counts.values())} line(s) were skipped.

```lackpy
for status in sorted(counts):
    print(status, counts[status])
```
