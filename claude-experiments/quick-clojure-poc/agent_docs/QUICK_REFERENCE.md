# LLDB Quick Reference Card

## Start Debugging

```bash
./debug.sh
# OR
lldb target/debug/examples/debug_spill
(lldb) run
```

## Essential Commands

| Command | Description |
|---------|-------------|
| `run` | Start program (will break at brk #0) |
| `si` | Step one instruction |
| `c` | Continue execution |
| `finish` | Run until current function returns |
| `bt` | Show backtrace/call stack |
| `q` | Quit lldb |

## Register Commands

| Command | Description |
|---------|-------------|
| `register read` | Show all registers |
| `register read x0 x1 x29 sp` | Show specific registers |
| `register write x29 0x12345678` | Set register value |

## Memory Commands

| Command | Description |
|---------|-------------|
| `memory read -fx -c16 $sp` | Show 16 quadwords at stack pointer |
| `memory read -fx $x29-24` | Show memory at x29-24 |
| `x/16gx $sp` | Alternative syntax (16 giant/8-byte hex) |

## Disassembly Commands

| Command | Description |
|---------|-------------|
| `disassemble -c 10` | Show next 10 instructions |
| `disassemble -s $pc -c 20` | Disassemble 20 from program counter |
| `di -c 5` | Shorthand for disassemble |

## Useful Calculations

| Command | Description |
|---------|-------------|
| `p/x $x29 - $sp` | Distance between x29 and sp (hex) |
| `p/d $x29 - $sp` | Distance between x29 and sp (decimal) |
| `p/x $x29 - 24` | Calculate spill slot 0 address |

## ARM64 Register Reference

| Registers | Purpose |
|-----------|---------|
| x0-x7 | Arguments and temporaries |
| x8 | Indirect result location |
| x9-x15 | Temporaries (caller-saved) |
| x16-x17 | Intra-procedure-call temporaries |
| x18 | Platform register |
| x19-x28 | Callee-saved (must preserve) |
| x29 | Frame pointer (FP) |
| x30 | Link register (LR) - return address |
| sp | Stack pointer |

## What to Look For

### 1. At Breakpoint (in trampoline)
```lldb
(lldb) register read x1
```
✓ x1 should contain JIT function address (e.g., 0x00000001...)

### 2. After "mov x29, sp" (in JIT prologue)
```lldb
(lldb) register read x29 sp
```
✓ x29 should equal sp
✓ Both should be valid addresses (0x00000001...)
✗ x29 = 0 means bug!

### 3. At First Spill Store
```lldb
(lldb) disassemble -c 1
# Should show: stur x<N>, [x29, #<offset>]
(lldb) register read x29
(lldb) p/x $x29 + <offset>
```
✓ x29 should still be valid (non-zero)
✓ x29 + offset should be valid stack address
✗ x29 = 0 means something corrupted it!

### 4. Stack Alignment Check
```lldb
(lldb) register read sp
```
✓ Should end in 0 (e.g., 0x...ff90, 0x...fe80)
✗ Ending in other values (4, 8, c) = misaligned!

## Trampoline Code Sequence

1. `brk #0` ← **You start here**
2. `stp x29, x30, [sp, #-16]!`
3. `stp x27, x28, [sp, #-16]!`
4. `stp x25, x26, [sp, #-16]!`
5. `stp x23, x24, [sp, #-16]!`
6. `stp x21, x22, [sp, #-16]!`
7. `stp x19, x20, [sp, #-16]!`
8. `mov x29, sp`
9. `blr x1` ← **Calls JIT function**

## JIT Function Prologue Sequence

1. `stp x29, x30, [sp, #-16]!` ← **Save frame and link**
2. `mov x29, sp` ← **CRITICAL: Sets frame pointer**
3. `sub sp, sp, #N` ← **Allocate spill space**

## Expected Stack Layout (after JIT prologue)

```
High address
+----------------+
| ...            |
+----------------+ ← x29 (frame pointer)
| saved x30 (LR) |  [x29 + 8]
| saved x29 (FP) |  [x29 + 0]
+----------------+ ← x29 - 16
| (padding/gap)  |
+----------------+ ← x29 - 24
| spill slot 0   |  [x29 - 24]
+----------------+ ← x29 - 32
| spill slot 1   |  [x29 - 32]
+----------------+ ← x29 - 40
| spill slot 2   |  [x29 - 40]
+----------------+ ← sp (stack pointer)
| ...            |
Low address
```

## Common Spill Offsets

| Slot | Offset | Address |
|------|--------|---------|
| 0 | -24 | [x29 - 24] |
| 1 | -32 | [x29 - 32] |
| 2 | -40 | [x29 - 40] |
| N | -24-8*N | [x29 - 24 - 8*N] |

## Debugging Checklist

- [ ] x1 contains valid JIT address at trampoline start
- [ ] Trampoline saves all registers correctly (check SP decreases by 96)
- [ ] Trampoline sets x29 = sp before calling JIT
- [ ] JIT function prologue saves x29, x30
- [ ] JIT function sets x29 = sp
- [ ] x29 is non-zero at first spill operation
- [ ] Spill offset matches expected formula: -24 - 8*slot
- [ ] Stack pointer stays 16-byte aligned

## Quick Test

```lldb
# Run to breakpoint
(lldb) run

# Check x1 has JIT address
(lldb) register read x1

# Step until you see "blr x1"
(lldb) si
(lldb) disassemble -c 1
# (repeat until you see blr)

# Step into JIT function
(lldb) si

# Step through prologue (3 instructions)
(lldb) si
(lldb) si
(lldb) si

# Check x29 is valid
(lldb) register read x29
# Should be non-zero!

# Continue until first stur
(lldb) disassemble -c 10
# (keep stepping with 'si' until you find stur)

# At stur, check x29 again
(lldb) register read x29
# Still non-zero? Good!
# Zero? Found the bug!
```

## File Reference

- **START_HERE.md** - Overview and getting started
- **LLDB_WALKTHROUGH.md** - Detailed step-by-step guide
- **DEBUG_GUIDE.md** - Architecture and debugging strategy
- **debug_commands.lldb** - Custom lldb command aliases
- **examples/debug_spill.rs** - Test program source
