# Profiling Setup for Pyret Parser

## Files Created

1. **large_test.arr** (108KB, 1,701 lines)
   - Auto-generated Pyret file for profiling
   - Contains:
     - 500 factorial functions
     - 500 fibonacci functions  
     - 500 object expressions
     - 200 check blocks

2. **Profiling build profile** in Cargo.toml
   - Custom `[profile.profiling]` section
   - Inherits from release (optimized code)
   - Includes debug symbols (`debug = true`)

## Building for Profiling

```bash
# Build with profiling profile (optimized + debug symbols)
cargo build --profile profiling

# The binary will be at:
target/profiling/pyret-attempt2
```

## Running Performance Tests

```bash
# Basic timing test
time target/profiling/pyret-attempt2 --mode json large_test.arr >/dev/null

# Current performance: ~0.05s for 1,701 lines
```

## Using with Profilers

### macOS Instruments

```bash
# Run with Instruments Time Profiler
instruments -t "Time Profiler" target/profiling/pyret-attempt2 --mode json large_test.arr

# Or use Xcode Instruments GUI
# File → Open → select target/profiling/pyret-attempt2
# Choose Time Profiler template
# Set arguments: --mode json large_test.arr
```

### Samply (Rust profiler)

```bash
# Install samply
cargo install samply

# Run profiler
samply record target/profiling/pyret-attempt2 --mode json large_test.arr

# Opens Firefox with flame graph
```

### perf (Linux)

```bash
# Record
perf record --call-graph=dwarf target/profiling/pyret-attempt2 --mode json large_test.arr

# View report
perf report

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

## Output Modes for Analysis

```bash
# Tokenize only (measure lexer performance)
time target/profiling/pyret-attempt2 --mode tokenize large_test.arr >/dev/null

# Parse only (measure parser performance)
time target/profiling/pyret-attempt2 --mode parse large_test.arr >/dev/null

# Full pipeline (tokenize + parse + JSON serialize)
time target/profiling/pyret-attempt2 --mode json large_test.arr >/dev/null
```

## File Statistics

- **Size:** 108KB
- **Lines:** 1,701
- **Parse time:** ~50ms (optimized build)
- **Output JSON:** ~MB

## Generating Larger Files

To create even larger test files for more intensive profiling:

```python
python3 -c "
# Adjust these numbers for larger files
NUM_FUNCTIONS = 2000
for i in range(NUM_FUNCTIONS):
    print(f'fun f_{i}(n): if n <= 0: 1 else: n * f_{i}(n - 1) end end')
print('\"done\"')
" > huge_test.arr
```

## Binary Information

```bash
# Check binary has debug symbols
file target/profiling/pyret-attempt2
# Output: Mach-O 64-bit executable arm64

# Check binary size
ls -lh target/profiling/pyret-attempt2
# Output: ~1.9M (with debug symbols)

# Verify it's optimized
cargo build --profile profiling --verbose
# Should show optimization flags
```
