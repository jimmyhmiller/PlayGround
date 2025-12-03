# Minimal Spill Bug Reproducer - Ready to Debug!

## Quick Start

```bash
# Build and run (will hang/crash)
./minimal_spill_test

# Debug with LLDB
lldb ./minimal_spill_test
```

## What's Included

- **minimal_spill_test.rs** - Standalone minimal test case (crashes with spilling)
- **MINIMAL_DEBUG_GUIDE.md** - Complete debugging guide
- **.vscode/launch.json** - VS Code debug configuration
- **lldb_init.txt** - LLDB initialization script

## The Bug in One Sentence

JIT code works perfectly with 10 registers (no spilling) but crashes/hangs when using 4 registers (forces spilling), despite all generated code being correct.

## Test Output

### Working (10 registers, no spills):
```
Compiling with 0 registers (default)...
✓ Compilation successful
Executing...
✓ Result: 15
✓✓✓ TEST PASSED! ✓✓✓
```

### Broken (4 registers, with spills):
```
Compiling with 4 registers (should force spilling)...
DEBUG: 1 spills, 1 total stack slots
✓ Compilation successful
Executing...
DEBUG: Calling trampoline.execute()...
DEBUG: About to call trampoline function...
[HANGS HERE]
```

## What's Been Verified

✅ All spill offsets are correct
✅ Stack allocation is correct
✅ ARM64 instruction encoding is correct
✅ The code works WITHOUT spilling
✅ Register allocation respects limits

## Debug It!

The minimal test is ready. Just run it with lldb and step through to find where it crashes/hangs.

Key suspect: Something subtle in how the JIT code executes (instruction cache, memory ordering, calling convention edge case, etc.)
