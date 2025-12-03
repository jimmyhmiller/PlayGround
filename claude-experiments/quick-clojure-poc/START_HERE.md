# LLDB Debugging Setup - Ready to Use!

## What's Been Set Up

I've added a breakpoint instruction (`brk #0`) at the start of the trampoline code in `src/trampoline.rs:141`. This will automatically trigger when you run the test program, allowing you to step through the code instruction by instruction.

## How to Start Debugging

### Option 1: Use the debug script (easiest)

```bash
./debug.sh
```

Then in lldb:
```lldb
(lldb) run
```

### Option 2: Manual

```bash
cargo build --example debug_spill
lldb target/debug/examples/debug_spill
```

In lldb:
```lldb
(lldb) run
```

## What Happens

1. The program builds and runs
2. It compiles this Clojure code: `(let [a 1 b 2 c 3 d 4 e 5] (+ a (+ b (+ c (+ d e)))))`
3. With only 4 registers available, it forces spilling
4. When execution reaches the trampoline, it hits `brk #0` and stops
5. You can now step through instruction by instruction with `si`

## Key Files

- **LLDB_WALKTHROUGH.md** - Detailed step-by-step debugging guide with what to look for
- **DEBUG_GUIDE.md** - Overview of the debugging process
- **debug_commands.lldb** - Useful command aliases for lldb
- **examples/debug_spill.rs** - The test program that triggers the bug

## Quick Reference - Essential LLDB Commands

```lldb
# Run the program (will stop at brk #0)
run

# Step one instruction
si

# Show key registers
register read x0 x1 x29 x30 sp

# Show stack memory
memory read -fx -c16 $sp

# Show current instruction
disassemble -c 1

# Continue execution
c
```

## What You're Looking For

The bug is that **x29 (frame pointer) appears to be 0 or corrupted when spill loads/stores execute**.

### The Critical Moment

After the JIT function prologue runs, you should see:
1. `stp x29, x30, [sp, #-16]!` - saves frame pointer
2. `mov x29, sp` - **THIS IS CRITICAL** - sets x29 to current sp
3. `sub sp, sp, #N` - allocates spill space

**After step 2, x29 MUST be valid (non-zero).** If it's 0, something went wrong.

### The Failing Point

Later, when you see a spill store:
```assembly
stur x<N>, [x29, #-24]
```

If this crashes or x29 is 0, you've found the bug!

## Debugging Strategy

1. **Step through trampoline** - verify x1 contains JIT function address
2. **Step through JIT prologue** - verify x29 gets set correctly
3. **Find first spill operation** - check if x29 is still valid
4. **If x29 is 0** - work backwards to find what corrupted it

## Example Session

```lldb
$ lldb target/debug/examples/debug_spill
(lldb) run

# You're now at brk #0 in the trampoline
(lldb) register read x1
# x1 should contain JIT function address

(lldb) si
(lldb) si
# ... keep stepping through trampoline ...

# Eventually you'll hit "blr x1" - step into it
(lldb) si

# Now you're in JIT code - step through prologue
(lldb) disassemble -c 10
(lldb) si    # stp x29, x30, [sp, #-16]!
(lldb) si    # mov x29, sp
(lldb) register read x29 sp
# x29 should equal sp now!

# Continue until you find a stur instruction
(lldb) disassemble -c 10
# ... keep stepping ...

# When you find: stur xN, [x29, #-24]
(lldb) register read x29
# If x29 is 0, you found the bug!
```

## Need Help?

See **LLDB_WALKTHROUGH.md** for a complete step-by-step guide with:
- What to check at each step
- Expected values for registers
- How to identify common issues
- Memory layout diagrams

Good luck debugging!
