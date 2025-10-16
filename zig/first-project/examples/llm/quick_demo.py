#!/usr/bin/env python3
"""
Quick demo - just compile, run, and show the output
"""
import subprocess
import tiktoken
import re
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Running GPT-2 generation...")
print("=" * 60)

# Run the Lisp implementation
result = subprocess.run(
    ["zig", "run", "../../src/simple_c_compiler.zig", "--", "llm.lisp", "--run"],
    capture_output=True,
    text=True,
    timeout=120
)

# Find and decode the tokens
enc = tiktoken.get_encoding("gpt2")
lines = result.stdout.split('\n')

for i, line in enumerate(lines):
    if "First 20 generated tokens:" in line:
        # Tokens are on next line
        if i + 1 < len(lines):
            token_line = lines[i + 1]
            match = re.search(r'\[([\d\s,]+)', token_line)
            if match:
                token_str = match.group(1)
                tokens = [int(t.strip()) for t in token_str.split(',') if t.strip().isdigit()]

                print("\nGenerated tokens:", tokens[:20])
                print("\nDecoded text:")
                print("-" * 60)

                # Decode all tokens at once for faster display
                text = enc.decode(tokens)
                print(text)

                print("-" * 60)
        break

print("\nDone!")
