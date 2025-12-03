# `let` Register Allocation Bug Fix

## The Bug

When implementing `let`, we discovered a critical register allocation bug:

```clojure
user=> (let [x 2 y 3] x)
3  ; WRONG! Should be 2

user=> (let [x 2 y 3 z 4 a1 2 a2 3 a3 4 a4 5 a6 7 a7 8] x)
8  ; WRONG! Should be 2
```

The pattern: **`let` always returned the value of the LAST BINDING instead of the body expression result**.

## Root Cause

The bug was in the **linear scan register allocator** in `src/register_allocation/linear_scan.rs`.

### What Was Happening

1. **Compilation was correct**:
   ```rust
   // For (let [x 2 y 3] x), the compiler generated:
   LoadConstant(v0, 2)  // x bound to virtual register 0
   LoadConstant(v1, 3)  // y bound to virtual register 1
   // Body: x compiles to v0 (correct!)
   // Result register: v0 (correct!)
   ```

2. **Register allocation was wrong**:
   ```
   v0: lifetime [0, 0]  // Only live at instruction 0
   v1: lifetime [1, 1]  // Only live at instruction 1

   v0 allocated to physical register p28
   v1 allocated to physical register p28  // REUSED THE SAME REGISTER!
   ```

3. **Execution**:
   ```assembly
   mov p28, #16    ; Load 2 (tagged) into p28
   mov p28, #24    ; Load 3 (tagged) into p28 - OVERWRITES x!
   mov x0, p28     ; Return p28 (contains 3, not 2!)
   ```

### Why This Happened

The register allocator computed lifetimes by scanning instructions:
- `v0` was defined at instruction 0 (LoadConstant)
- `v0` was never *used* in any instruction (the body just returns it)
- So the allocator thought v0's lifetime was `[0, 0]` - it dies immediately!

The allocator then **reused physical register p28** for v1, since v0 appeared to be dead.

But v0 **wasn't actually dead** - it was the **result register** of the entire expression!

## The Fix

We added `LinearScan::mark_live_until_end()` to extend the lifetime of result registers.

### Changes

**File**: `src/register_allocation/linear_scan.rs`

Added method to extend lifetimes:
```rust
/// Mark a register as live until the end (for result registers)
pub fn mark_live_until_end(&mut self, register: VirtualRegister) {
    let end_index = self.instructions.len().saturating_sub(1);
    if let Some((start, _)) = self.lifetimes.get(&register) {
        self.lifetimes.insert(register, (*start, end_index));
    } else {
        // Register not seen - make it live for the whole function
        self.lifetimes.insert(register, (0, end_index));
    }
}
```

**File**: `src/arm_codegen.rs`

Call the method before running allocation:
```rust
// Run linear scan register allocation
let mut allocator = LinearScan::new(instructions.to_vec(), 0);

// Mark result register as live until the end
// This is critical - without this, the register allocator may reuse
// the physical register for the result, causing wrong values to be returned
if let IrValue::Register(vreg) = result_reg {
    allocator.mark_live_until_end(*vreg);
}

allocator.allocate();
```

### After The Fix

```
v0: lifetime [0, 1]  // Extends to the end!
v1: lifetime [1, 1]

v0 allocated to physical register p28
v1 allocated to physical register p27  // Different register!
```

Now the execution is correct:
```assembly
mov p28, #16    ; Load 2 into p28 (x)
mov p27, #24    ; Load 3 into p27 (y)
mov x0, p28     ; Return p28 (contains 2!) ✅
```

## Test Results

All tests now pass:

```clojure
(let [x 2] x)                                    => 2 ✅
(let [x 2 y 3] x)                                => 2 ✅
(let [x 2 y 3 z 4] x)                            => 2 ✅
(let [x 2 y 3 z 4 a1 2 a2 3 a3 4 a4 5 a6 7 a7 8] x) => 2 ✅
```

## Lessons Learned

1. **Register allocation is subtle** - even when compilation is correct, wrong lifetime analysis breaks everything

2. **Result registers need special handling** - they're implicitly live until function return, even if not referenced in instructions

3. **Testing matters** - the original tests didn't catch this because they used single bindings where register reuse didn't matter

4. **Debug carefully** - we added debug output to see:
   - What the compiler was generating (correct)
   - What the register allocator was doing (wrong)
   - This pinpointed the exact issue

## Impact

This fix is **critical** for all code that uses `let`:
- Without it, any `let` with multiple bindings would return garbage
- Functions (which will use `let`-style scoping) would be broken
- Closures (which capture locals) would capture wrong values

**The bug affected the entire compilation pipeline, not just `let`.**

## Status

✅ Bug fixed
✅ All 15 let tests passing
✅ Original user test case working
✅ Ready for `fn` implementation (which depends on correct local variable handling)
