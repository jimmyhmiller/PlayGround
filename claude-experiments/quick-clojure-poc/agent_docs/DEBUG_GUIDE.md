# Debugging JIT Code with LLDB

This guide shows how to step through the JIT code instruction by instruction to debug the spilling issue.

## Quick Start

```bash
# Build the debug example
cargo build --example debug_spill

# Run with lldb
lldb target/debug/examples/debug_spill
```

## LLDB Commands

Once in lldb:

```lldb
# Load helpful command aliases
(lldb) command source debug_commands.lldb

# Run the program - it will hit the breakpoint in the trampoline
(lldb) run

# Now you're stopped at the brk #0 instruction in the trampoline
# Step through instructions one at a time
(lldb) si

# Show register state
(lldb) register read
(lldb) register read x0 x1 x29 x30 sp

# Show stack memory (32 quadwords starting at SP)
(lldb) memory read -fx -c32 $sp

# Show memory at frame pointer
(lldb) memory read -fx -c32 $x29

# Disassemble current location
(lldb) disassemble -c 20

# Continue execution
(lldb) c

# If it crashes, see where:
(lldb) bt
```

## Custom Aliases (after loading debug_commands.lldb)

- `show-state` - Show key registers (x0-x11, x29, x30, sp, pc)
- `show-stack` - Show 32 quadwords of stack memory
- `show-frame` - Show 32 quadwords at frame pointer
- `si-show` - Step one instruction and show registers

## What to Look For

### 1. Trampoline Entry
- Check that x1 contains the JIT function address
- Watch x29 (frame pointer) and sp (stack pointer) as registers are saved

### 2. Before BLR to JIT Code
- Note the values of x29 and sp
- These should be valid stack addresses
- x29 should point to the frame the trampoline set up

### 3. JIT Function Prologue
- Watch as the JIT function saves x29 and x30
- Check that x29 is set to sp after the save
- If there's stack allocation (sub sp, sp, #N), verify the amount

### 4. First Spill Store
- Find the first STUR instruction (store to frame)
- Check the offset: should be [x29, #offset] where offset is negative
- For slot 0: should be [x29, #-24]
- **CRITICAL**: Verify x29 is not 0 or invalid at this point!

### 5. Spill Load
- Find LDUR instructions (load from frame)
- Verify they're loading from the same offsets as stores
- Check that the loaded values match what was stored

## Expected Stack Layout in JIT Code

After JIT prologue runs:
```
[x29 + 0]:  saved x29, x30 (from STP instruction)
[x29 - 16]: (start of spill area after SUB sp, sp, #space)
[x29 - 24]: spill slot 0
[x29 - 32]: spill slot 1
[x29 - 40]: spill slot 2
...
```

## Trampoline Stack Layout

The trampoline saves these in order:
1. x29, x30 (frame pointer and link register)
2. x19-x20
3. x21-x22
4. x23-x24
5. x25-x26
6. x27-x28

Total: 96 bytes (6 pairs × 16 bytes/pair)

## Common Issues to Debug

1. **x29 is 0 or invalid when accessing spills**
   - The prologue didn't run correctly
   - The frame pointer wasn't set up

2. **Spill offset calculation wrong**
   - Check the STUR/LDUR immediate values
   - Should be negative offsets from x29

3. **Stack pointer misaligned**
   - ARM64 requires 16-byte alignment
   - Check that SP is always 0x...0 (last hex digit is 0)

4. **Wrong calling convention**
   - Verify trampoline passes JIT address in x1
   - Verify JIT prologue doesn't corrupt x1 before using it
