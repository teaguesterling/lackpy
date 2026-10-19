from pathlib import Path

path = Path("missing.txt")
try:
    text = path.read_text()
except FileNotFoundError:
    text = "absent"
print(text)
