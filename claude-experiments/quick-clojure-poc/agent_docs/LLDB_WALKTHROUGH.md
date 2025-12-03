# LLDB Walkthrough: Debugging JIT Spilling Issue

## Quick Start

```bash
# Build and start debugging
cargo build --example debug_spill
lldb target/debug/examples/debug_spill

# In lldb, run the program
(lldb) run
```

The program will automatically hit a breakpoint (`brk #0`) at the start of the trampoline code.

## Step-by-Step Debugging Instructions

### 1. Start at the Trampoline Breakpoint

When the program hits the `brk #0` instruction, you're at the start of the trampoline.

```lldb
(lldb) register read x0 x1 x29 x30 sp
```

**What to check:**
- `x0` = unused (first argument, ignored)
- `x1` = address of the JIT function (this is the critical value!)
- `x29` = current frame pointer (from calling function)
- `x30` = return address (where to return after trampoline finishes)
- `sp` = stack pointer (should be 16-byte aligned: ends in 0x...0)

**Expected:** x1 should contain a valid pointer address (e.g., 0x000000010abcd000)

### 2. Step Through Trampoline Prologue

```lldb
(lldb) si
(lldb) disassemble -c 1
```

The trampoline saves registers in this order:

1. **Save x29, x30** - frame pointer and link register
   ```
   stp x29, x30, [sp, #-16]!
   ```
   After: SP -= 16

2. **Save x27, x28** through **x19, x20** - callee-saved registers (5 pairs)
   ```
   stp x27, x28, [sp, #-16]!
   stp x25, x26, [sp, #-16]!
   stp x23, x24, [sp, #-16]!
   stp x21, x22, [sp, #-16]!
   stp x19, x20, [sp, #-16]!
   ```
   After: SP -= 80 (total 96 bytes allocated)

3. **Set up frame pointer**
   ```
   mov x29, sp
   ```
   After: x29 = current SP (points to saved x19, x20)

**Check after each STP:**
```lldb
(lldb) si
(lldb) register read sp x29
(lldb) memory read -fx -c4 $sp
```

### 3. Call to JIT Function

```lldb
# Step to the BLR instruction
(lldb) si
(lldb) disassemble -c 1
```

You should see:
```
blr x1
```

**Before executing this:**
```lldb
(lldb) register read x1 x29 sp
(lldb) memory read -fx -c16 $sp
```

**Critical check:**
- `x1` = JIT function address (should match value from step 1)
- `x29` = points to trampoline's frame (where x19-x20 were saved)
- `sp` = same as x29 (frame pointer points to top of saved registers)

**Now step INTO the JIT function:**
```lldb
(lldb) si
(lldb) disassemble -c 5
```

### 4. JIT Function Prologue

You should now see the JIT function's prologue:

```assembly
stp x29, x30, [sp, #-16]!    ; Save frame pointer and link register
mov x29, sp                   ; Set up new frame pointer
sub sp, sp, #<N>              ; Allocate stack space for spills (if needed)
```

**Step through each instruction:**

#### a. After STP (save x29, x30):
```lldb
(lldb) si
(lldb) register read x29 sp
(lldb) memory read -fx -c8 $sp
```

**Check:**
- SP should have decreased by 16
- Memory at [SP] should contain old x29
- Memory at [SP+8] should contain old x30 (return address to trampoline)

#### b. After MOV (set frame pointer):
```lldb
(lldb) si
(lldb) register read x29 sp
```

**Critical check:**
- `x29` should now equal `sp`
- `x29` should point to where we just saved the old frame pointer
- **This x29 value is what spill loads/stores will use as their base!**

#### c. After SUB (allocate spill space):
```lldb
(lldb) si
(lldb) register read x29 sp
(lldb) p/d $x29 - $sp
```

**Check:**
- SP should have decreased by the allocation size
- x29 should still point to the saved frame pointer
- The distance between x29 and sp should match the allocation size

**Example with 3 spills:**
- Allocation: `sub sp, sp, #32` (24 bytes for 3 spills + 8 bytes padding)
- x29 - sp should equal 32

### 5. Find First Spill Store

Continue stepping until you find a `STUR` (store unscaled register) instruction:

```lldb
(lldb) si
(lldb) disassemble -c 1
```

Look for:
```assembly
stur x<N>, [x29, #<offset>]
```

**Before executing:**
```lldb
(lldb) register read x29 x<N>
(lldb) memory read -fx -c8 $x29
```

**Critical checks:**
- **x29 must not be 0!** If x29 is 0, the prologue didn't run correctly
- x29 should point to the saved frame (where we stored old x29/x30)
- The offset should be negative (e.g., #-24 for slot 0)

**Calculate target address:**
```lldb
(lldb) p/x $x29 + <offset>
```

For slot 0, offset = -24:
```lldb
(lldb) p/x $x29 - 24
```

This should give a valid stack address (should be < x29 and >= sp).

**Execute the store:**
```lldb
(lldb) si
(lldb) memory read -fx $x29-24
```

The value should now be stored at [x29 - 24].

### 6. Find Spill Load

Continue stepping until you find a `LDUR` (load unscaled register) instruction:

```assembly
ldur x<N>, [x29, #<offset>]
```

**Before executing:**
```lldb
(lldb) register read x29
(lldb) memory read -fx $x29-24
```

**Check:**
- x29 should still be valid (not 0)
- Memory at [x29 + offset] should contain the value we stored earlier

**Execute the load:**
```lldb
(lldb) si
(lldb) register read x<N>
```

The register should now contain the value from memory.

### 7. Common Issues and How to Spot Them

#### Issue: x29 is 0 or invalid

```lldb
(lldb) register read x29
x29 = 0x0000000000000000
```

**Diagnosis:** The JIT prologue didn't execute correctly.

**Check:**
1. Did we actually execute the `mov x29, sp` instruction?
2. Was SP valid when we executed it?
3. Did something overwrite x29 after we set it?

#### Issue: Wrong spill offset

```lldb
# Expected: stur x0, [x29, #-24]
# Actual: stur x0, [x29, #-80]
```

**Diagnosis:** Spill offset calculation is wrong.

**Formula should be:** [x29 - 16 - ((slot + 1) * 8)]
- slot 0: x29 - 24
- slot 1: x29 - 32
- slot 2: x29 - 40

#### Issue: Stack pointer misaligned

```lldb
(lldb) register read sp
sp = 0x000000016fdff8c4    # Ends in 4, not 0!
```

**Diagnosis:** Stack allocations must be multiples of 16.

**Check:** In the prologue, the `sub sp, sp, #N` should have N as a multiple of 16.

### 8. Useful LLDB Commands Summary

```lldb
# Execution control
si                          # Step one instruction
c                           # Continue execution
finish                      # Run until current function returns

# Examining state
register read               # Show all registers
register read x0 x1 x29 sp  # Show specific registers
disassemble -c 10           # Show next 10 instructions
bt                          # Backtrace (show call stack)

# Memory examination
memory read -fx -c16 $sp    # Show 16 quadwords at stack pointer
memory read -fx $x29-24     # Show memory at x29-24
p/x $x29 - $sp              # Calculate distance between x29 and sp

# Disassembly
disassemble -s $pc -c 20    # Disassemble 20 instructions from PC
disassemble -n main         # Disassemble named function
```

### 9. Expected Flow Summary

1. **Trampoline breakpoint** → x1 contains JIT function address
2. **Trampoline saves registers** → SP decreases by 96 bytes
3. **Trampoline sets x29** → x29 points to saved registers
4. **BLR to JIT function** → x30 set to trampoline return address
5. **JIT saves x29, x30** → SP decreases by 16
6. **JIT sets x29 = sp** → x29 now points to JIT's frame
7. **JIT allocates stack** → SP decreases by spill space
8. **Spill stores use [x29, #-offset]** → Negative offsets from x29
9. **Spill loads use [x29, #-offset]** → Same offsets
10. **JIT epilogue** → Restore and return
11. **Trampoline epilogue** → Restore callee-saved registers
12. **Return to Rust** → With result in x0

## Debugging Strategy

1. **First run:** Step through everything, verify x29 is always valid
2. **If x29 is 0:** Focus on the JIT prologue (steps 5-7)
3. **If offsets wrong:** Check the spill offset calculation in the code
4. **If alignment wrong:** Check stack allocation size in prologue
5. **If loads fail:** Verify stores happened first with correct offsets

## Pro Tips

- Set a condition on PC to stop at specific addresses
- Use `memory write` to fix values and continue
- Use `register write x29 <value>` to manually fix x29
- Print instruction encodings with `x/4xb $pc`
