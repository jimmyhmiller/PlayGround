#!/usr/bin/env python3
"""
Simple tokenizer wrapper using tiktoken for GPT-2
Provides encoding (text -> tokens) and decoding (tokens -> text)
"""
import sys
import tiktoken

def encode(text):
    """Encode text to token IDs"""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return tokens

def decode(token_id):
    """Decode a single token ID to text"""
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode([token_id])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Encode: python tokenize.py encode 'your text here'")
        print("  Decode: python tokenize.py decode 12345")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "encode":
        text = " ".join(sys.argv[2:])
        tokens = encode(text)
        # Print tokens space-separated for easy parsing
        print(" ".join(map(str, tokens)))

    elif mode == "decode":
        token_id = int(sys.argv[2])
        text = decode(token_id)
        # Print raw text without newline
        print(text, end="", flush=True)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
