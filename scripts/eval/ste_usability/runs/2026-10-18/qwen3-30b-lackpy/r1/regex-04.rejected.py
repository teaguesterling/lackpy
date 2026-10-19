from pathlib import Path

# /// intent
# goal   = "Print the names of the functions defined in the .cpp files under src (the word before '(' on lines that start with 'int '), sorted."
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

paths = Path("src").glob("*.cpp")
function_names = set()

for p in paths:
    text = p.read_text()
    lines = text.splitlines()
    for line in lines:
        if line.strip().startswith("int "):
            line = line.strip()
            start = line.find('(')
            if start != -1:
                # Extract the function name: the token before '('
                # Split the line and find the last token before '('
                tokens = line[:start].split()
                if len(tokens) > 1:
                    name = tokens[-1]
                    # Remove any trailing non-alphanumeric characters (like '&', 'const')
                    while name and not name[-1].isalnum():
                        name = name[:-1]
                    if name:
                        function_names.add(name)

# Convert to sorted list
sorted_names = sorted(function_names)

# Print each function name
for name in sorted_names:
    print(name)
