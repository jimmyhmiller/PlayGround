# Minimal Spill Test - Debugging Guide

## The Bug

The JIT compiler works perfectly WITHOUT spilling (using 10 registers), but **crashes/hangs when spilling is enabled** (using 4 registers).

## What's Been Fixed

✅ Register allocation now respects the register limit parameter
✅ Spill offset calculation is correct (slot N → offset -(N+1)*8)
✅ Stack space allocation is correct
✅ Instruction encoding is correct

## The Minimal Test Case

`minimal_spill_test.rs` - A standalone program that:
1. Creates simple IR: loads 5 constants, untags them, adds them (expects result: 15)
2. Compiles with 4 registers - forces 1 spill
3. Executes the generated code
4. **Hangs/crashes** during execution

## How to Debug

### Option 1: Run the test

```bash
./minimal_spill_test
```

**Expected:** Hangs after "About to call trampoline function..."

### Option 2: Debug with LLDB

```bash
lldb ./minimal_spill_test
(lldb) breakpoint set -n quick_clojure_poc::trampoline::Trampoline::execute
(lldb) run
```

When it breaks:
```lldb
(lldb) register read
(lldb) frame variable self jit_fn
(lldb) continue
```

The program will hang after continuing.

### Option 3: VS Code

1. Open the project in VS Code
2. Use the "Debug Minimal Spill Test" launch configuration
3. Set breakpoints as needed
4. Press F5

## What to Look For

### The Generated Code

The test outputs the machine code. Key instructions to examine:

**Prologue:**
```
0x0000: stp x29, x30, [sp, #-16]!  - Save FP/LR
0x0004: mov x29, sp                 - Set FP
0x0008: sub sp, sp, #N              - Allocate stack (if spills exist)
```

**Spill Store:**
```
0x00XX: stur xN, [x29, #-offset]   - Store to stack
```

**Spill Load:**
```
0x00YY: ldur xN, [x29, #-offset]   - Load from stack
```

### Common Issues to Check

1. **Is x29 valid when spill operations execute?**
   - x29 should point to the saved FP/LR location
   - x29 should NOT be 0

2. **Are spill offsets within allocated stack space?**
   - Check that all offsets are above SP
   - Formula: [x29 + offset] should be >= SP

3. **Is the trampoline corrupting registers?**
   - The trampoline saves x19-x28
   - Check if x9-x11 (temp registers) are being corrupted

4. **Is there an infinite loop in the generated code?**
   - Disassemble and check for unexpected branches

## Stack Layout (When Working Correctly)

After JIT prologue with 2 spills (16 bytes + 8 padding = 24 bytes):

```
High Address
+------------------+
| Caller's frame   |
+------------------+ ← Original SP
| saved x30 (LR)   | [x29 + 8]
| saved x29 (FP)   | [x29 + 0] ← x29 points here
+------------------+
| spill slot 0     | [x29 - 8]
| spill slot 1     | [x29 - 16]
+------------------+ ← x29 - 24 ← SP points here
Low Address
```

## The Mystery

Everything appears correct in the generated code:
- Offsets are mathematically correct
- Encoding is valid ARM64
- Stack allocation is sufficient
- The code works WITHOUT spilling

But it still crashes/hangs WITH spilling. The bug must be subtle - possibly:
- Instruction cache not being flushed properly
- Memory alignment issue
- Calling convention violation
- Race condition or timing issue
- Something in how the trampoline calls the JIT code

## Next Steps

Use LLDB to:
1. Verify x29 is set correctly in the JIT prologue
2. Check that spill stores actually write to valid memory
3. Verify spill loads read back the correct values
4. Look for any unexpected behavior in register values

The crash/hang happens somewhere between "About to call trampoline function..." and the function returning.
