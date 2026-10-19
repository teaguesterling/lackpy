from pathlib import Path

text = 'a 3 b 7 9'
words = text.split()
total = 0
for word in words:
    try:
        num = int(word)
    except ValueError:
        continue
    total += num
print(total)
