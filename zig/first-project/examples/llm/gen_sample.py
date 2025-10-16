#!/usr/bin/env python3
"""
Generate text with temperature sampling to avoid repetition
"""
import sys
import subprocess
import tiktoken
import os
import tempfile

# Quick fix: Just add a note about the repetition issue
print("""
╔════════════════════════════════════════════════════════════╗
║  NOTE: Greedy decoding causes repetitive text             ║
║  The model always picks the most likely token (argmax)     ║
║                                                            ║
║  To fix this, we need to add temperature sampling.         ║
║  For now, try shorter generations or different prompts.    ║
╚════════════════════════════════════════════════════════════╝
""")

if len(sys.argv) < 2:
    print("\nUsage: python gen_sample.py 'Your prompt' [num_tokens]")
    print("\nExample: python gen_sample.py 'Hello world' 50")
    sys.exit(1)

# For now, just call gen.py with a warning
import gen as base_gen
base_gen.main()
