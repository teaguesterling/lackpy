"""Write docs/reference/style-guide.md from the rule registry and the dictionary.

Run with the project interpreter: .venv/bin/python scripts/gen_style_guide.py
"""

from pathlib import Path

from lackpy.lang.spec import format_style_guide

OUT = Path(__file__).resolve().parents[1] / "docs" / "reference" / "style-guide.md"

if __name__ == "__main__":
    OUT.write_text(format_style_guide())
    print(f"wrote {OUT}")
