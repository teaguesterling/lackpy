# /// intent
# goal   = "Convert each word in the text 'a 3 b 7 9' to an integer if possible, skip invalid ones, and print their sum"
# reads  = []
# writes = []
# output = "printed"
# ///

from pathlib import Path

text = "a 3 b 7 9"
total = 0
for word in text.split():
    try:
        num = int(word)
    except ValueError:
        continue
    total += num
print(total)
