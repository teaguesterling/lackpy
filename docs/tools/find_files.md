# find_files

Find files matching a glob pattern.

## Signature

```python
find_files(pattern: str) -> list[str]
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `pattern` | `str` | Glob pattern (e.g. `"*.py"`, `"src/**/*.toml"`) |

## Returns

A sorted list of matching file paths as strings, relative to the workspace root.

## Grade

- **World coupling (w):** 1 — reads filesystem metadata
- **Effects ceiling (d):** 1 — no side effects

## Examples

```python
py_files = find_files("*.py")
```

```python
all_configs = find_files("**/*.toml")
```

## Notes

- Uses `pathlib.Path.glob()` under the hood.
- Returns an empty list if no files match.
- Results are sorted lexicographically.
- Recursive patterns use `**` (e.g. `"**/*.md"`).
