# Interactive GPT-2 Text Generation

This directory contains interactive demos for GPT-2 text generation using both the Lisp implementation and the original C implementation.

## Setup

1. **Create and activate Python virtual environment:**
   ```bash
   cd examples/llm
   python3 -m venv venv
   source venv/bin/activate
   pip install tiktoken
   ```

2. **Make sure you have the GPT-2 model file:**
   ```bash
   # Should exist at: gpt2_124M.bin
   ls -lh gpt2_124M.bin
   ```

## Usage

### Interactive Demo (Recommended)

Shows token generation with real-time decoding to text:

```bash
source venv/bin/activate
python interactive.py "Hello world" lisp
```

**Arguments:**
- First argument: Your text prompt (required)
- Second argument: Implementation to use - `lisp`, `c`, or `both` (default: `both`)

**Examples:**
```bash
# Run Lisp implementation only
python interactive.py "The quick brown fox" lisp

# Run original C implementation only
python interactive.py "Once upon a time" c

# Run both implementations for comparison
python interactive.py "In a galaxy far far away"
```

### Tools

- **`interactive.py`** - Main demo script with nice formatting and token decoding
- **`tokenize.py`** - Utility for encoding/decoding individual tokens:
  ```bash
  python tokenize.py encode "Hello world"   # -> token IDs
  python tokenize.py decode 15496           # -> text
  ```

## How It Works

1. **Prompt Encoding**: Your text is encoded to GPT-2 token IDs using `tiktoken`
2. **Generation**: The model runs forward passes to generate new tokens
3. **Decoding**: Each generated token is decoded back to text in real-time
4. **Streaming Display**: Tokens are displayed one at a time for a typewriter effect

## Current Limitations

- The implementations currently use hardcoded prompts ("Hello world is")
- Custom prompt support will be added in a future update
- The demo shows generation with the default prompt regardless of your input

## Performance

The Lisp implementation uses an optimized matrix multiplication with:
- Loop unrolling by 8
- Register tiling
- Weight reuse (8x memory bandwidth reduction)

This achieves performance within ~2x of the original C implementation.

## Implementation Details

### Lisp Implementation (`llm.lisp`)
- Fixed context window (8 tokens) for O(N) generation
- Optimized matmul with loop unrolling
- Generates 100 tokens
- Compiled with `-O3` optimization

### Original C Implementation (`original_llm_c/`)
- Reference implementation from llm.c
- Same optimizations as Lisp version
- Used for performance comparison

## Files

- `interactive.py` - Main interactive demo
- `tokenize.py` - Token encoding/decoding utility
- `llm.lisp` - Lisp GPT-2 implementation
- `generate_tokens.lisp` - Minimal token generation example
- `venv/` - Python virtual environment (created on setup)
- `README_INTERACTIVE.md` - This file

## Next Steps

Future improvements:
- [ ] Add custom prompt support (pass tokens via stdin)
- [ ] Add temperature and top-k sampling
- [ ] Stream tokens in real-time during generation
- [ ] Add OpenMP multi-threading support
- [ ] Create web interface for easier interaction
