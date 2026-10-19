# /// intent
# goal   = "Convert each word in the text 'a 3 b 7 9' to an integer if possible, skip non-integer words, and print the sum of the valid integers"
# reads  = []
# writes = []
# output = "printed"
# ///

import json
from pathlib import Path

# Read the input text
text = "a 3 b 7 9"

# Convert each word to integer if possible, skip others
total = 0
for word in text.split():
    try:
        total += int(word)
    except ValueError:
        # Skip non-integer words
        continue

# Print the sum
print(total)
