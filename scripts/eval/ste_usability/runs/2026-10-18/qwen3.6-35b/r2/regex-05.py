# /// intent
# goal   = "Print the number of lines in logs/app.log whose level is WARN or ERROR."
# reads  = ["logs/app.log"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

LOGS = Path("logs/app.log")

def count_bad_levels(text: str) -> int:
    lines = text.split("\n")
    count = 0
    for line in lines:
        if " ERROR " in line or " WARN " in line:
            count += 1
    return count

def main() -> None:
    text = LOGS.read_text()
    n = count_bad_levels(text)
    print(n)

main()
