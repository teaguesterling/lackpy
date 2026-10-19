# /// intent
# goal   = "Convert words in 'a 3 b 7 9' to integers where possible and print the sum"
# reads  = []
# writes = []
# output = "printed"
# ///

def to_int(s):
    try:
        return int(s)
    except ValueError:
        return None

text = 'a 3 b 7 9'
words = text.split()

total = 0
for w in words:
    n = to_int(w)
    if n is not None:
        total += n

print(total)
