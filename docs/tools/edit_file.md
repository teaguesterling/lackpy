# edit_file

Replace the first occurrence of a string in a file.

## Signature

```python
edit_file(path: str, old_str: str, new_str: str) -> bool
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `path` | `str` | File path relative to the workspace root |
| `old_str` | `str` | The text to find and replace |
| `new_str` | `str` | The replacement text |

## Returns

`True` if the replacement was made, `False` if `old_str` was not found.

## Grade

- **World coupling (w):** 3 — writes to the filesystem
- **Effects ceiling (d):** 3 — modifies existing files

## Examples

```python
edit_file("config.toml", 'debug = false', 'debug = true')
```

```python
success = edit_file("main.py", "old_function()", "new_function()")
```

## Notes

- Only replaces the **first** occurrence of `old_str`.
- Returns `False` without modifying the file if `old_str` is not found.
- The match is exact (case-sensitive, whitespace-sensitive).
- Including this tool in a kit raises the kit grade to w=3, d=3.
