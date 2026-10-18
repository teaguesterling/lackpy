# /// intent
# goal   = "Convert each word to int where possible, skip others, and print the sum"
# reads  = []
# writes = []
# output = "printed"
# ///

text = 'a 3 b 7 9'
total = 0
for word in text.split():
    try:
        n = int(word)
        total = total + n
    except ValueError:
        continue
print(total)
