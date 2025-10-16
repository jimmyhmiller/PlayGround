#!/usr/bin/env python3
"""
Generate text with custom prompts
Generates a custom .lisp file with your prompt tokens embedded
"""
import sys
import subprocess
import tiktoken
import os
import tempfile

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen.py 'Your prompt text' [num_tokens]")
        print("Example: python gen.py 'Hello world' 50")
        sys.exit(1)

    prompt = sys.argv[1]
    num_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)

    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens}")
    print(f"Generating {num_tokens} new tokens...")
    print()

    # Read the base llm.lisp file
    with open('llm.lisp', 'r') as f:
        base_code = f.read()

    # Find the hardcoded prompt section and replace it
    # Look for the lines:
    #   (pointer-index-write! sequence 0 15496)  ; "Hello"
    #   (pointer-index-write! sequence 1 995)    ; " world"
    #   (pointer-index-write! sequence 2 318)    ; " is"

    # Generate replacement code
    token_init_code = "\n".join([
        f'        (pointer-index-write! sequence {i} {token_id})'
        for i, token_id in enumerate(tokens)
    ])

    # Also update the initial count
    # Find: (let [gen_count (: I32) 0
    #           total_generated (: I32) 3]  ; Start with 3 prompt tokens
    # Replace total_generated with len(tokens)

    # Replace the hardcoded sections
    custom_code = base_code.replace(
        '''        ;; Initialize prompt tokens
        (pointer-index-write! sequence 0 15496)  ; "Hello"
        (pointer-index-write! sequence 1 995)    ; " world"
        (pointer-index-write! sequence 2 318)    ; " is"

        (printf (c-str "Initial prompt tokens: [15496, 995, 318]\\n"))''',
        f'''        ;; Initialize prompt tokens (generated from: {prompt})
{token_init_code}

        (printf (c-str "Initial prompt tokens: {tokens}\\n"))'''
    ).replace(
        'total_generated (: I32) 3]  ; Start with 3 prompt tokens',
        f'total_generated (: I32) {len(tokens)}]  ; Start with {len(tokens)} prompt tokens'
    ).replace(
        'num_tokens_to_generate (: I32) 100',
        f'num_tokens_to_generate (: I32) {num_tokens}'
    ).replace(
        '''        ;; Print final sequence (first 20 tokens for readability)
        (printf (c-str "\\nFirst 20 generated tokens:\\n"))
        (printf (c-str "["))
        (let [i (: I32) 0
              print_limit (: I32) (if (< (+ 3 num_tokens_to_generate) 20)
                                    (+ 3 num_tokens_to_generate)
                                    20)]''',
        f'''        ;; Print ALL generated tokens
        (printf (c-str "\\nAll generated tokens:\\n"))
        (printf (c-str "["))
        (let [i (: I32) 0
              print_limit (: I32) (+ {len(tokens)} num_tokens_to_generate)]'''
    )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lisp', delete=False) as f:
        temp_file = f.name
        f.write(custom_code)

    try:
        # Compile and run
        result = subprocess.run(
            ['zig', 'run', '../../src/simple_c_compiler.zig', '--', temp_file, '--run'],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output
        for line in result.stdout.split('\n'):
            if 'All generated tokens:' in line:
                idx = result.stdout.split('\n').index(line)
                token_line = result.stdout.split('\n')[idx + 1]
                # Extract tokens
                import re
                match = re.search(r'\[([\d\s,]+)', token_line)
                if match:
                    token_str = match.group(1)
                    gen_tokens = [int(t.strip()) for t in token_str.split(',') if t.strip().isdigit()]

                    print(f"\nGenerated {len(gen_tokens)} total tokens (prompt + new)")
                    print(f"Token IDs: {gen_tokens}")
                    print()

                    # Decode ALL tokens
                    text = enc.decode(gen_tokens)
                    print("Generated text:")
                    print("=" * 60)
                    print(text)
                    print("=" * 60)
                break
    finally:
        # Clean up temp file
        os.unlink(temp_file)

if __name__ == "__main__":
    main()
