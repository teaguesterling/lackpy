# /// intent
# goal   = "Print the names of functions defined in .cpp files under src, sorted alphabetically"
# reads  = ["src/a.cpp", "src/b.cpp", "src/c.cpp"]
# writes = []
# output = "printed"
# ///

import re
from pathlib import Path

# Extract function names from .cpp files in src directory
function_names = set()

# Pattern to match lines that start with 'int ' followed by a function name and '('
pattern = re.compile(r'^int\s+(\w+)\s*\(')

# Iterate over all .cpp files in src directory
for cpp_file in Path('src').glob('*.cpp'):
    try:
        content = cpp_file.read_text()
        for line in content.splitlines():
            match = pattern.match(line.strip())
            if match:
                function_names.add(match.group(1))
    except (OSError, UnicodeDecodeError) as e:
        # Handle file reading errors: OSError for I/O issues, UnicodeDecodeError for encoding
        # Catch specific exceptions instead of Exception
        pass

# Sort the function names and print them
for name in sorted(function_names):
    print(name)
