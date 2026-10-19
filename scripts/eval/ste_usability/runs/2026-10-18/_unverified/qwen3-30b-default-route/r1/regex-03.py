# /// intent
# goal   = "Replace every 'code=XNN' pattern in logs/app.log with 'code=[XNN]' and write the result to logs/app.marked.log"
# reads  = ["logs/app.log"]
# writes = ["logs/app.marked.log"]
# output = "printed"
# ///

import re
from pathlib import Path

# Read the log file
log_path = Path("logs/app.log")
if not log_path.exists():
    raise FileNotFoundError("logs/app.log not found")

# Read the content
text = log_path.read_text()

# Replace code=XNN with code=[XNN
# X is a letter, NN are digits
pattern = r'code=([A-Z])(\d{2})'
replacement = r'code=[\1\2]'
marked_text = re.sub(pattern, replacement, text)

# Write the result to the marked log file
marked_log_path = Path("logs/app.marked.log")
marked_log_path.write_text(marked_text)

# Print the result
print(marked_text)
