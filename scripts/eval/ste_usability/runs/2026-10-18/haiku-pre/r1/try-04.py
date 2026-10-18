# /// intent
# goal   = "Convert each word of the text 'a 3 b 7 9' to an integer where possible, skip the others, and print the sum."
# reads  = []
# writes = []
# output = "printed"
# ///

text = 'a 3 b 7 9'
total = 0
for word in text.split():
    try:
        num = int(word)
    except ValueError:
        continue
    total = total + num
print(total)
