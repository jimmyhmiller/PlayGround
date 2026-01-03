# Lispier Project Instructions

## CRITICAL: Never Use /dev/stdin in Piped Commands

**NEVER EVER RUN A SCRIPT WITH THIS PATTERN:**

```bash
echo '...' | cargo run --quiet -- show-ast /dev/stdin 2>&1
```

The `/dev/stdin` will make the entire Claude Code session lock up and become unrecoverable!

### Why This Happens

`/dev/stdin` is a symlink to `/dev/fd/0` (the process's stdin file descriptor). When a child process opens `/dev/stdin`, it accesses the shell's stdin (the TTY). Claude Code communicates with the shell via this same TTY, causing a deadlock.

### Safe Alternatives

Instead of piping to `/dev/stdin`, write to a temporary file:

```bash
# WRONG - will hang:
echo '(module ...)' | cargo run --quiet -- show-ast /dev/stdin

# CORRECT - use a temp file:
echo '(module ...)' > /tmp/test.lisp && cargo run --quiet -- show-ast /tmp/test.lisp

# CORRECT - if testing files that already exist:
cargo run --quiet -- show-ast examples/simple.lisp
```

If a hang occurs, you must kill the Claude Code process entirely.

## Remote GPU Testing

The `scripts/remote-test.sh` script syncs code to the GPU server and runs GPT-2 inference.

### Usage

```bash
./scripts/remote-test.sh
```

### What It Does

1. Syncs local files to `jimmyhmiller@computer.jimmyhmiller.com:~/Code/lispier`
2. Builds the project with `cargo build --release`
3. Runs GPT-2 GPU inference: `cargo run --release -- run examples/gpt2/gpt2_generate_gpu.lisp`

### Server Details

- Primary host: `192.168.0.55` (local network)
- Backup host: `computer.jimmyhmiller.com`
- GPU: AMD gfx1151 (ROCm backend)
- Model: GPT-2 124M (~124M parameters)
- Performance: ~450ms per token

### Running Directly on Server

```bash
ssh jimmyhmiller@computer.jimmyhmiller.com
source ~/.cargo/env
cd lispier
cargo run --release -- run examples/gpt2/gpt2_generate_gpu.lisp
```
