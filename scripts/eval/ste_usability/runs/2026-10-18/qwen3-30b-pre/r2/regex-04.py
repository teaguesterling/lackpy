# /// intent
# goal   = "Print the names of functions defined in .cpp files under src, sorted"
# reads  = ["src/*.cpp"]
# writes = []
# output = "printed"
# ///

from pathlib import Path

# Find all .cpp files in src directory
paths = Path("src").glob("*.cpp")

# List to store function names
function_names = []

# Process each .cpp file
for p in sorted(paths):
    text = p.read_text()
    # Extract lines that start with 'int ' and contain '('
    for line in text.splitlines():
        if line.strip().startswith("int ") and "(" in line:
            # Extract the function name (word after 'int ')
            # Split by space and take the part before '('
            name_part = line.strip().split()[1]
            # Remove parentheses if present
            if "(" in name_part:
                name = name_part[:name_part.index("(")]
            else:
                name = name_part
            function_names.append(name)

# Sort function names alphabetically
function_names.sort()

# Print each function name
for name in function_names:
    print(name)
