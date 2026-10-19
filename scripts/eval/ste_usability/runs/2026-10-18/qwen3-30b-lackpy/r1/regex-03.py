from pathlib import Path
import re

# Read the input file
text = Path("logs/app.log").read_text()

# Replace every 'code=XNN' with 'code=[XNN]'
# where X is a letter and NN are digits
pattern = r'code=([A-Za-z][0-9]{2})'
replacement = r'code=[\1]'
marked_text = re.sub(pattern, replacement, text)

# Write the result to the output file
Path("logs/app.marked.log").write_text(marked_text)
