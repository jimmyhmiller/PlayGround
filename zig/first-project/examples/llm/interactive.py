#!/usr/bin/env python3
"""
Interactive GPT-2 Demo - Real-time token generation with decoding
Streams generated tokens one at a time, decoded to text

Make sure to activate the virtual environment first:
  source venv/bin/activate
  python interactive.py 'Your prompt'
"""
import sys
import subprocess
import tiktoken
import os
import time
import re

def main():
    if len(sys.argv) < 2:
        print("╔════════════════════════════════════════════════════════╗")
        print("║   Interactive GPT-2 Text Generation                   ║")
        print("╚════════════════════════════════════════════════════════╝")
        print()
        print("Usage: python interactive.py 'Your prompt here' [implementation]")
        print()
        print("Arguments:")
        print("  prompt         - Text to autocomplete (required)")
        print("  implementation - 'lisp' or 'c' (default: both)")
        print()
        print("Examples:")
        print("  python interactive.py 'Hello world'")
        print("  python interactive.py 'The quick brown fox' lisp")
        print("  python interactive.py 'Once upon a time' c")
        print()
        sys.exit(1)

    prompt = sys.argv[1]
    impl = sys.argv[2].lower() if len(sys.argv) > 2 else "both"

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║              GPT-2 Text Generation Demo               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"Prompt: \"{prompt}\"")
    print()

    # Encode prompt
    prompt_tokens = enc.encode(prompt)
    print(f"Encoded to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print()

    # Note: The current implementations have hardcoded prompts
    # This will be improved to accept custom prompts
    print("Note: Current implementations use hardcoded prompt.")
    print("      Showing generation with default prompt...")
    print()

    if impl in ["lisp", "both"]:
        print("─" * 60)
        print(" LISP IMPLEMENTATION")
        print("─" * 60)
        print()

        script_dir = os.path.dirname(os.path.abspath(__file__))

        print("Compiling and running... (this may take a moment)")
        print()

        # Compile and run
        result = subprocess.run(
            [
                "zig", "run",
                os.path.join(script_dir, "../../src/simple_c_compiler.zig"),
                "--",
                os.path.join(script_dir, "llm.lisp"),
                "--run"
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=script_dir
        )

        # Parse output for generated tokens
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if "First 20 generated tokens:" in line:
                # The tokens are on the next line
                if i + 1 < len(lines):
                    token_line = lines[i + 1]
                    # Extract tokens from [token1, token2, ...] format
                    match = re.search(r'\[([\d\s,]+)', token_line)
                    if match:
                        token_str = match.group(1)
                        # Split by comma and parse each number
                        tokens = []
                        for t in token_str.split(','):
                            t = t.strip()
                            if t and t.isdigit():
                                tokens.append(int(t))

                        print("Generated text:")
                        print("─" * 60)
                        print()

                        # Decode tokens one by one for streaming effect
                        for token_id in tokens:
                            text = enc.decode([token_id])
                            print(text, end="", flush=True)
                            time.sleep(0.05)  # Small delay for effect

                        print()
                        print()
                break

        # Show timing if available
        for line in result.stderr.split('\n'):
            if 'real' in line or 'user' in line:
                print(f"  {line.strip()}")

        print()

    if impl in ["c", "both"]:
        print("─" * 60)
        print(" ORIGINAL C IMPLEMENTATION")
        print("─" * 60)
        print()

        orig_dir = "/Users/jimmyhmiller/Documents/Code/PlayGround/zig/first-project/original_llm_c"

        print("Running original C implementation...")
        print()

        # Run C implementation
        result = subprocess.run(
            ["./bench_gen"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=orig_dir
        )

        # Parse output for generated tokens
        for line in result.stdout.split('\n'):
            if "First 20 tokens:" in line:
                # Extract tokens from "First 20 tokens: 1234 5678 ..." format
                parts = line.split(':')[1].strip().split(' ')
                tokens = []
                for p in parts:
                    p = p.strip()
                    if p and p.isdigit():
                        tokens.append(int(p))

                if tokens:
                    print("Generated text:")
                    print("─" * 60)
                    print()

                    # Decode tokens one by one for streaming effect
                    for token_id in tokens:
                        text = enc.decode([token_id])
                        print(text, end="", flush=True)
                        time.sleep(0.05)  # Small delay for effect

                    print()
                    print()
                break

        # Show timing if available
        for line in result.stdout.split('\n'):
            if 'Total time:' in line or 'Per token:' in line:
                print(f"  {line.strip()}")

        print()

    print("╚════════════════════════════════════════════════════════╝")
    print()

if __name__ == "__main__":
    main()
