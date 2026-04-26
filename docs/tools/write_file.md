# write_file

Write content to a file, creating or overwriting it.

## Signature

```python
write_file(path: str, content: str) -> bool
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `path` | `str` | File path relative to the workspace root |
| `content` | `str` | Full text content to write |

## Returns

`True` on success.

## Grade

- **World coupling (w):** 3 — writes to the filesystem
- **Effects ceiling (d):** 3 — creates or overwrites files

## Examples

```python
write_file("output.txt", "hello world")
```

```python
lines = ["line 1", "line 2", "line 3"]
write_file("data.txt", "\n".join(lines))
```

## Notes

- Overwrites the file if it already exists.
- Creates parent directories only if they exist; does not create intermediate directories.
- Including this tool in a kit raises the kit grade to w=3, d=3.
