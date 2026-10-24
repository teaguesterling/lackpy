# People

Read the rows, then drop the header and the empty lines, one step per cell.

```lackpy
rows = "name,n\nann,3\nbob,5\ncat,7\n\n".splitlines()
```

The raw text has {len(rows)} lines.

```lackpy
rows = rows[1:]
```

```lackpy
rows = [r for r in rows if r]
print(len(rows))
```

There are {len(rows)} people.

```lackpy
items = ["a", "b", "c", "d"]
```

```lackpy
items = items[1:]
```

```lackpy
items = items[1:]
print(items)
```
