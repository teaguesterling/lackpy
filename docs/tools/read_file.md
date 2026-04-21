# read_file

Read the contents of a file as a string.

## Signature

```python
read_file(path: str) -> str
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `path` | `str` | File path relative to the workspace root |

## Returns

The full text content of the file.

## Grade

- **World coupling (w):** 1 — reads from the filesystem
- **Effects ceiling (d):** 1 — no side effects

## Examples

```python
content = read_file("README.md")
```

```python
config = read_file("src/config.toml")
lines = config.split("\n")
```

## Notes

- Raises `FileNotFoundError` if the path does not exist.
- Reads the entire file into memory; not suitable for very large files.
- Path is resolved relative to the workspace directory at execution time.
