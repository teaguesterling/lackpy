# Default: 0.5b picks tools, qwen2.5:1.5b assembles (our winner), with baseline
../venv/bin/python scripts/pluckit-quartermaster.py --host http://localhost:11435 --baseline --timeout 60 --output qm-default.json

# Try smollm2 as assembler
../venv/bin/python scripts/pluckit-quartermaster.py --host http://localhost:11435 --asm-model smollm2 --baseline --timeout 60 --output qm-smollm2.json

# Try phi4-mini as assembler
../venv/bin/python scripts/pluckit-quartermaster.py --host http://localhost:11435 --asm-model phi4-mini --timeout 60 --output qm-phi4.json

# Try llama3.2:1b as both QM and assembler (smallest viable?)
../venv/bin/python scripts/pluckit-quartermaster.py --host http://localhost:11435 --qm-model llama3.2:1b --asm-model llama3.2:1b --timeout 60

# Try qwen2.5:1.5b as its own QM (can the winner do both?)
../venv/bin/python scripts/pluckit-quartermaster.py --host http://localhost:11435 --qm-model qwen2.5:1.5b --asm-model qwen2.5:1.5b --baseline --timeout 60


