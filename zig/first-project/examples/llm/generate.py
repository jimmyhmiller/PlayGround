#!/usr/bin/env python3
"""
Interactive GPT-2 text generation
Supports both the original C implementation and the Lisp implementation
"""
import sys
import subprocess
import tiktoken
import os

def encode_text(text):
    """Encode text to GPT-2 token IDs"""
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text)

def decode_token(token_id):
    """Decode a single token ID to text"""
    enc = tiktoken.get_encoding("gpt2")
    try:
        return enc.decode([token_id])
    except:
        return f"<ERR:{token_id}>"

def run_generation_lisp(prompt, num_tokens=50):
    """Run generation using the Lisp implementation"""
    print(f"\n{'='*60}")
    print(f"LISP IMPLEMENTATION")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Generating {num_tokens} tokens...")
    print(f"{'='*60}\n")

    # Encode the prompt
    tokens = encode_text(prompt)
    print(f"Encoded to {len(tokens)} tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}\n")

    # Create a modified Lisp file that accepts command line input
    # For now, let's just run the existing implementation and parse its output
    # We'll compile and run llm.lisp which generates 100 tokens

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Output: ", end="", flush=True)
    print(prompt, end="", flush=True)

    # Run the Lisp implementation
    # Since we can't easily pass custom prompts yet, we'll use the hardcoded one
    result = subprocess.run(
        ["zig", "run", "../../src/simple_c_compiler.zig", "--", "llm.lisp", "--run"],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Parse the output to extract generated tokens
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if "First 20 generated tokens:" in line:
            # Extract tokens from the line
            token_line = line.split('[')[1].split(']')[0]
            token_strs = token_line.split(',')

            # Decode and print each token
            for token_str in token_strs[:num_tokens]:
                try:
                    token_id = int(token_str.strip())
                    text = decode_token(token_id)
                    print(text, end="", flush=True)
                except:
                    pass
            break

    print("\n" + "="*60 + "\n")

def run_generation_c(prompt, num_tokens=50):
    """Run generation using the original C implementation"""
    print(f"\n{'='*60}")
    print(f"ORIGINAL C IMPLEMENTATION")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Generating {num_tokens} tokens...")
    print(f"{'='*60}\n")

    # Encode the prompt
    tokens = encode_text(prompt)
    print(f"Encoded to {len(tokens)} tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}\n")

    # Change to original_llm_c directory
    orig_dir = "/Users/jimmyhmiller/Documents/Code/PlayGround/zig/first-project/original_llm_c"
    os.chdir(orig_dir)

    print("Output: ", end="", flush=True)
    print(prompt, end="", flush=True)

    # Run the C implementation
    result = subprocess.run(
        ["./bench_gen"],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Parse the output to extract generated tokens
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if "First 20 tokens:" in line:
            # Extract tokens from the line
            parts = line.split(':')[1].strip().split(' ')

            # Decode and print each token
            for token_str in parts[:num_tokens]:
                try:
                    token_id = int(token_str)
                    text = decode_token(token_id)
                    print(text, end="", flush=True)
                except:
                    pass
            break

    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Interactive GPT-2 Text Generation")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python generate.py '<your prompt>' [num_tokens] [implementation]")
        print()
        print("Arguments:")
        print("  prompt         - Text to complete (required)")
        print("  num_tokens     - Number of tokens to generate (default: 50)")
        print("  implementation - 'lisp', 'c', or 'both' (default: 'both')")
        print()
        print("Examples:")
        print("  python generate.py 'Hello world'")
        print("  python generate.py 'The quick brown fox' 30 lisp")
        print("  python generate.py 'Once upon a time' 100 c")
        sys.exit(1)

    prompt = sys.argv[1]
    num_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    impl = sys.argv[3].lower() if len(sys.argv) > 3 else "both"

    if impl in ["lisp", "both"]:
        run_generation_lisp(prompt, num_tokens)

    if impl in ["c", "both"]:
        run_generation_c(prompt, num_tokens)

if __name__ == "__main__":
    main()
