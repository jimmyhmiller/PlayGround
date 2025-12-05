# Closure Implementation Documentation

## Overview

This document describes how closures with captured values are implemented in our Clojure→ARM64 JIT compiler. The implementation follows a traditional closure representation where captured variables are stored in a heap-allocated closure object.

## Register Usage and Calling Conventions

### ARM64 Calling Convention (Standard C ABI)

- **x0-x7**: Argument registers (first 8 arguments)
- **x8**: Special purpose register
- **x9-x15**: Temporary/caller-saved registers
- **x16-x17**: Intra-procedure-call temporary registers
- **x18**: Platform register (reserved)
- **x19-x28**: Callee-saved registers (preserved across function calls)
- **x29**: Frame pointer (FP)
- **x30**: Link register (LR) - stores return address
- **x31/sp**: Stack pointer

### Our Function Calling Conventions

We use TWO different calling conventions depending on whether the function has closures:

#### 1. Regular Functions (No Closures)
- Arguments passed in **x0-x7**
- No special closure object
- Function represented as a tagged code pointer: `(code_ptr << 3) | 0b100`
- NOT a heap object - just a tagged pointer to executable code

#### 2. Closures (With Captured Variables)
- Arguments passed in **x0-x7**
- **x8** contains the tagged closure object pointer
- Closure is a heap object with tag `0b101`
- The closure object contains both the code pointer and captured values

### Virtual Register Allocation Strategy

To avoid conflicts between argument registers and compiler-generated virtual registers:

- **Virtual registers v0-v8 with `is_argument: true`**: Reserved for function parameters and closure object
  - v0-v7: Function arguments (map to x0-x7)
  - v8: Closure object parameter (maps to x8)

- **Virtual registers starting at v9**: Compiler-generated temporaries
  - The `IrBuilder::new()` sets `next_register = 9`
  - This prevents conflicts where we might have both:
    - `VirtualRegister { index: 8, is_argument: true }` (closure parameter)
    - `VirtualRegister { index: 8, is_argument: false }` (regular temp)

### Register Allocation (Linear Scan)

- **Pre-allocation phase**: Argument registers (v0-v8 with `is_argument: true`) are pre-allocated to their corresponding physical registers (x0-x8)

- **Linear scan phase**:
  - Uses callee-saved registers **x19-x28** (10 registers by default)
  - Argument registers are filtered out - they don't participate in linear scan
  - When saving registers across calls, we use callee-saved registers to avoid corruption

- **Physical registers excluded from linear scan pool**:
  - x0-x8: Reserved for arguments and closure object
  - x9-x15: Could be used but currently not in the pool
  - x29: Frame pointer
  - x30: Link register
  - x31: Stack pointer

## Closure Object Layout

Closures are heap-allocated objects with the following layout:

```
Offset  | Size | Field
--------|------|------------------
0       | 8    | Header (GC metadata)
8       | 8    | Field 0: name_ptr (string pointer or 0)
16      | 8    | Field 1: code_ptr (untagged)
24      | 8    | Field 2: closure_count
32      | 8    | Field 3: closure_value[0]
40      | 8    | Field 4: closure_value[1]
...     | ...  | ...
```

**Important**: When accessing closure values via LoadClosure:
- The closure pointer in x8 is TAGGED (0b101)
- We must untag (shift right by 3) before accessing fields
- Closure value at index N is at byte offset: `32 + (N * 8)`

## How Closures Work (Step by Step)

### Example Code
```clojure
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add-five (make-adder 5))
(add-five 3)  ; Should return 8
```

### Step 1: Compiling the Outer Function `(fn [x] ...)`

**IR Generated:**
```
Label L0              ; Entry point for outer function
  Jump L3
Label L2              ; Entry point for inner function
  LoadClosure(v1, v8, 0)  ; Load captured 'x' from closure object in v8
  Untag(v3, v1)           ; Untag captured x
  Untag(v4, v0)           ; Untag argument y
  AddInt(v5, v3, v4)      ; Add x + y
  Tag(v6, v5, 0)          ; Tag result
  Ret(v6)
Label L3
  MakeFunction(v11, L2, [v0])  ; Create closure, capturing v0 (parameter x)
  Ret(v11)
```

**What happens:**
- The outer function takes parameter `x` in v0 (maps to x0)
- It creates a closure that captures `x`
- `MakeFunction(v11, L2, [v0])` allocates a heap object containing:
  - Code pointer to L2 (the inner function)
  - The value of v0 (the captured `x`)
- Returns the tagged closure pointer

### Step 2: Calling `(make-adder 5)`

**Code generation for MakeFunction:**
```arm64
; Save closure values to callee-saved registers FIRST
; (before setting up x0-x2, to avoid overwriting them)
mov x19, x0              ; Save captured value (parameter x=40, which is 5<<3)

; Set up arguments for trampoline_allocate_function
mov x0, #0               ; name_ptr (anonymous)
adr x1, L2               ; code_ptr (address of inner function)
mov x2, #1               ; closure_count = 1

; Move saved closure value to argument position
mov x3, x19              ; First closure value

; Call trampoline (with X30 preservation)
sub sp, sp, #16
str x30, [sp]
mov x15, <trampoline_allocate_function address>
blr x15
ldr x30, [sp]
add sp, sp, #16

; Result in x0 is the tagged closure pointer
mov x28, x0              ; Move to destination register
```

**Trampoline allocates closure:**
- Creates heap object with closure_count=1
- Stores code_ptr=L2, closure_values[0]=40 (5<<3)
- Tags pointer with 0b101 and returns

### Step 3: Compiling `(add-five 3)`

**IR Generated:**
```
LoadVar(v5, <add-five var>)        ; Load closure from var
LoadConstant(v6, 24)                ; Constant 3 (tagged as 24 = 3<<3)
Call(v7, v5, [v6])                  ; Call closure with argument
```

**Call instruction codegen:**
```arm64
; Step 1: Save function and arguments to callee-saved registers
mov x19, x28             ; Save function pointer (closure)
mov x21, x27             ; Save argument (3<<3)

; Step 2: Extract tag from function value
and x10, x19, #0b111     ; x10 = tag

; Step 3: Check if Function (0b100) or Closure (0b101)
cmp x10, #0b100
b.eq is_function_label

; === Closure path (tag == 0b101) ===
; Untag closure pointer
lsr x11, x19, #3         ; x11 = untagged closure pointer

; Load code_ptr from closure field 1 (offset 16)
ldr x20, [x11, #16]      ; x20 = code pointer

; Set up closure calling convention
mov x8, x19              ; x8 = tagged closure pointer
mov x0, x21              ; x0 = argument

; Jump past function path
b after_call_label

; === Function path (tag == 0b100) ===
is_function_label:
lsr x20, x19, #3         ; x20 = code pointer (untag)
mov x0, x21              ; x0 = argument

; === Call the function ===
after_call_label:
blr x20                  ; Call function

; Result in x0
mov x26, x0              ; Move result to destination
```

### Step 4: Executing the Inner Function

**When the inner function at L2 executes:**

1. **x0** contains argument `y` = 24 (3<<3)
2. **x8** contains tagged closure pointer

**LoadClosure instruction:**
```arm64
; LoadClosure(v1, v8, 0) - load closure value 0 from v8
; v8 is pre-allocated to x8 (the closure parameter)

; Untag the closure pointer
lsr x11, x8, #3          ; x11 = untagged closure pointer

; Load closure value from offset 32 (field 3)
ldr x28, [x11, #32]      ; x28 = closure_values[0] = 40 (5<<3)
```

3. **Computation:**
   - Untag captured x: 40 >> 3 = 5
   - Untag argument y: 24 >> 3 = 3
   - Add: 5 + 3 = 8
   - Tag result: 8 << 3 = 64
   - Return 64

4. **Return to caller:**
   - Result 64 in x0
   - Untagged by caller: 64 >> 3 = 8
   - Final result: **8** ✓

## Critical Implementation Details

### 1. X30 (Link Register) Preservation

When making external calls (trampolines), we must preserve X30 because:
- BLR (branch with link to register) overwrites X30 with the return address
- If we don't save X30 before the trampoline call, we lose our return address
- This caused infinite loops/hangs before the fix

**Solution:**
```arm64
sub sp, sp, #16      ; Allocate stack space
str x30, [sp]        ; Save X30
; ... call trampoline ...
ldr x30, [sp]        ; Restore X30
add sp, sp, #16      ; Deallocate
```

### 2. Closure Value Passing Order

When setting up arguments for `trampoline_allocate_function`:
- Must save closure values to callee-saved registers (x19-x23) FIRST
- Then set up x0-x2 (name_ptr, code_ptr, count)
- Then move from x19-x23 to x3-x7 (closure value arguments)

**Why?** If a closure value is in x0, and we do:
```arm64
mov x3, x0       ; Save closure value
mov x0, #0       ; Set up name_ptr - OVERWRITES x0!
```
We lose the closure value! By using callee-saved registers as intermediate storage, we avoid this.

### 3. LoadClosure Offset Calculation

The closure values start at field 3, which is at byte offset 32:
- Header: 8 bytes (offset 0)
- Field 0 (name_ptr): 8 bytes (offset 8)
- Field 1 (code_ptr): 8 bytes (offset 16)
- Field 2 (closure_count): 8 bytes (offset 24)
- Field 3 (closure_values[0]): 8 bytes (offset 32) ← First value!

**Bug we fixed:** Originally used offset 24, which was field 2 (the count), not the value!

### 4. Register Allocation Conflict

**Critical bug:** The compiler was creating both:
- `VirtualRegister { index: 8, is_argument: true }` - closure parameter
- `VirtualRegister { index: 8, is_argument: false }` - regular temp

These are DIFFERENT virtual registers (different `is_argument` flag), but HashMap treats them as different keys. During register allocation:
- v8 (is_argument: true) pre-allocated to x8
- v8 (is_argument: false) allocated to x26 by linear scan
- When replacing registers in IR, v8 gets replaced with x26, breaking closures!

**Fix:** Start virtual register numbering at 9 instead of 0, so regular temps never conflict with argument register indices.

## Trampoline Functions

Closures use several trampoline functions to interact with the GC runtime:

### `trampoline_allocate_function`
**Purpose:** Allocate a closure object on the heap

**ARM64 Calling Convention:**
- x0 = name_ptr (0 for anonymous)
- x1 = code_ptr (untagged address of function code)
- x2 = closure_count (number of captured values)
- x3-x7 = closure values (up to 5 values supported)
- **Returns:** x0 = tagged closure pointer (0b101)

### `trampoline_function_code_ptr`
**Purpose:** Get code pointer from closure (not currently used - we do this inline)

### `trampoline_function_get_closure`
**Purpose:** Get closure value by index (not currently used - we use LoadClosure inline)

## Code Organization

### Files Modified/Created

1. **src/ir.rs**
   - Added `MakeFunction` instruction
   - Added `LoadClosure` instruction
   - Changed `IrBuilder::new()` to start `next_register` at 9

2. **src/compiler.rs**
   - Closure analysis to find free variables
   - Generate MakeFunction IR with captured values
   - Create closure parameter (v8) for functions with free vars
   - Generate LoadClosure IR to access captured values

3. **src/arm_codegen.rs**
   - `MakeFunction` codegen (lines 534-629)
   - `LoadClosure` codegen (lines 631-655)
   - `Call` instruction with inline tag checking (lines 656-756)
   - X30 preservation around trampoline calls

4. **src/gc_runtime.rs**
   - `allocate_function` - creates closure heap objects
   - `function_code_ptr` - extracts code pointer from closure
   - `function_get_closure` - extracts closure values
   - Uses Closure tag (0b101) instead of generic HeapObject

5. **src/trampoline.rs**
   - `trampoline_allocate_function` - C ABI wrapper for allocate_function
   - Receives up to 5 closure values in x3-x7
   - Returns tagged closure pointer in x0

6. **src/register_allocation/linear_scan.rs**
   - Pre-allocates v0-v8 with `is_argument: true` to x0-x8
   - Filters argument registers from linear scan
   - Uses x19-x28 as allocatable physical registers

## Testing

### Test Cases (tests/test_closures.clj)

```clojure
; Test 1: Simple value capture
(def get-x (fn [x] (fn [] x)))
(def get-five (get-x 5))
(get-five)  ; Returns: 5 ✓

; Test 2: Arithmetic with captured value
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add-five (make-adder 5))
(add-five 3)  ; Returns: 8 ✓

; Test 3: Complex arithmetic with captured value
(def make-multiplier (fn [x] (fn [y] (+ (* x y) x))))
(def times-three-plus (make-multiplier 3))
(times-three-plus 4)  ; Returns: 15 (3*4 + 3) ✓
```

All tests pass successfully!

## Performance Considerations

### What's Fast

1. **Regular functions (no closures):** Very fast
   - No heap allocation
   - Just a tagged code pointer
   - Direct call via BLR after untagging

2. **Inline tag checking:**
   - No trampoline calls to determine calling convention
   - Just AND + CMP + branch to select path
   - Much faster than the previous approach

### What's Slower

1. **Closure allocation:**
   - Requires heap allocation (GC overhead)
   - Trampoline call to allocate_function
   - X30 save/restore overhead

2. **Closure calls:**
   - Extra memory access to load code pointer from heap
   - Extra register (x8) used for closure object

3. **LoadClosure:**
   - Memory access to fetch captured value from heap
   - Untag operation before access

### Optimization Opportunities (Not Implemented)

1. **Inline closure allocation** - avoid trampoline call
2. **Closure elimination** - detect when closures don't escape and use stack allocation
3. **Direct closure calls** - when we know statically it's a closure, skip tag check
4. **Register windows** - use more than 10 physical registers if needed

## Reflections and Design Questions

### What Works Well

1. **Separation of concerns:**
   - IR layer is clean and architecture-independent
   - Codegen handles ARM64-specific details
   - GC runtime manages heap objects

2. **Pre-allocation strategy:**
   - Argument registers are cleanly separated from temps
   - No spilling of argument registers during linear scan

3. **Inline tag checking:**
   - Fast dispatch between regular functions and closures
   - No runtime function calls just to determine calling convention

### What's Awkward

1. **Virtual register numbering:**
   - Having to start at 9 to avoid conflicts feels hacky
   - The `is_argument` flag distinction is subtle and easy to miss
   - Could we have separate namespaces for argument vs temp registers?

2. **Closure value limit:**
   - Currently limited to 5 captured values (x3-x7)
   - What if a closure captures more than 5 values?
   - Should we use stack or a different calling convention?

3. **X30 preservation is manual:**
   - Every external call needs manual save/restore
   - Easy to forget and cause subtle bugs
   - Could this be automated by the codegen?

4. **LoadClosure offset calculation:**
   - Magic number 32 is fragile
   - If heap layout changes, this breaks
   - Should there be a constant or function to compute this?

5. **Two calling conventions:**
   - Regular functions vs closures use different conventions
   - Increases complexity of Call codegen
   - But necessary for performance (regular functions shouldn't pay closure overhead)

### Alternative Designs to Consider

1. **Unified calling convention:**
   - ALL functions could be closures (even with 0 captured vars)
   - Simpler code generation
   - But slower for simple functions

2. **Closure environments as separate objects:**
   - Function object points to environment object
   - More flexible (can share environments)
   - But adds indirection

3. **Multiple closure calling conventions:**
   - Small closures (≤5 values) use x3-x7
   - Large closures use pointer to value array in x3
   - More complex but handles unlimited captures

4. **Lambda lifting / closure conversion:**
   - Convert closures to top-level functions with extra parameters
   - No heap allocation needed
   - But changes semantics (can't return closures easily)

### Questions for Discussion

1. **Is the virtual register numbering strategy acceptable?**
   - Starting at 9 works but feels like a workaround
   - Should we have explicit "temp register allocator" vs "argument register allocator"?

2. **Should we support more than 5 captured values?**
   - Current limit is arbitrary based on available registers
   - Real code might need more
   - What's the best way to handle this?

3. **Is the tag-based dispatch the right approach?**
   - It's fast, but adds complexity
   - Could we use vtables or other mechanisms?

4. **Should LoadClosure stay as an IR instruction?**
   - Currently it's first-class in the IR
   - Could it be lowered to simpler operations earlier?

5. **Do we need better separation between "function value" and "closure value"?**
   - Currently they're both IrValue::Register
   - Could distinct types help prevent bugs?

6. **Is the trampoline approach for allocation the right choice?**
   - Adds overhead but keeps GC code separate
   - Could we inline the allocation code?
   - What about making the GC API more JIT-friendly?

---

## Honest Reflection on the Implementation

Looking at what we've built, here's an honest assessment of the current design:

### What I Actually Like

1. **The tag-based dispatch is clever and fast.** Checking 3 bits to decide between regular functions and closures is much better than calling trampoline functions. This was a good architectural choice that came from studying Beagle.

2. **The IR abstraction holds up well.** Having `MakeFunction` and `LoadClosure` as first-class IR instructions makes the compiler logic clean. The IR doesn't need to know about ARM64 details.

3. **Pre-allocating argument registers makes sense.** It's conceptually clean that v0-v7 map to x0-x7, and v8 maps to x8 for closures.

### What Feels Wrong

1. **The virtual register numbering hack (starting at 9) is embarrassing.** This is a band-aid fix for a deeper design flaw. We have two namespaces (argument registers vs temps) but they share the same index space, distinguished only by a boolean flag. This led to the subtle bug where `VirtualRegister { index: 8, is_argument: true }` and `VirtualRegister { index: 8, is_argument: false }` were both valid but conflicting. **I don't like this solution** - it works, but it's fragile and someone could easily break it by accident in the future.

   **Better design would be:**
   ```rust
   enum VirtualRegister {
       Temp(usize),      // v_temp_0, v_temp_1, ...
       Argument(usize),  // v_arg_0, v_arg_1, ...
   }
   ```
   This makes the distinction explicit and prevents index conflicts.

2. **The 5-value closure limit is arbitrary and limiting.** What happens when someone writes code that captures 10 variables? Right now it just fails. We should either:
   - Support unlimited captures by passing a pointer to an array
   - Make it clear this is a prototype limitation
   - Or actually fix it properly

3. **Manual X30 preservation is error-prone.** Every time we add a new trampoline call, we have to remember to wrap it in save/restore. I already forgot once and it caused a hang. The codegen should handle this automatically - maybe detect external calls and inject the save/restore?

   **Could be:**
   ```rust
   impl Arm64CodeGen {
       fn emit_external_call(&mut self, target: usize) {
           // Automatically saves X30, calls, restores
       }
   }
   ```

4. **The LoadClosure offset calculation (32) is a magic number.** If someone changes the heap object layout, this will break silently. We should have:
   ```rust
   const CLOSURE_HEADER_SIZE: usize = 8;
   const CLOSURE_FIELDS_BEFORE_VALUES: usize = 3;  // name, code, count
   const CLOSURE_VALUES_OFFSET: usize = CLOSURE_HEADER_SIZE + (CLOSURE_FIELDS_BEFORE_VALUES * 8);
   ```
   At least then it's documented and computed from the structure.

5. **Having two calling conventions increases complexity.** The `Call` instruction codegen is now much more complex because it has to handle both regular functions and closures. Every time we modify calling convention, we have to update both paths. But I'm not sure this is avoidable - regular functions shouldn't pay the closure overhead.

### What I'm Uncertain About

1. **Should closures always be heap-allocated?**
   - Pro: Simple, straightforward, matches traditional closure implementation
   - Con: Slow, puts pressure on GC, can't be stack-allocated even when safe
   - Alternative: Escape analysis to detect when closures don't escape, then stack-allocate them
   - But that's way more complex...

2. **Is the IR the right level of abstraction for MakeFunction?**
   - Currently `MakeFunction(dst, label, [closure_values])`
   - The label is code-generation specific (it's literally an address in the compiled code)
   - Should the IR be more abstract? Maybe `MakeFunction(dst, function_id, [closure_values])` and resolve the label later?

3. **Should we use a different design for the closure object in x8?**
   - Currently we pass the TAGGED closure pointer in x8
   - Then we have to untag it every time we access it
   - Could we pass the UNTAGGED pointer? (But then how do we know it's a pointer for GC?)
   - Or use a different register entirely?

4. **The trampoline approach feels heavyweight.**
   - We're crossing from JIT code → trampoline → Rust → GC
   - Multiple function call overhead just to allocate
   - Could we generate allocation code inline? But then how do we track GC roots?

### Questions I'd Ask if This Were a Real Project

1. **Is the complexity worth it?** We have:
   - Two types of function values (regular vs closure)
   - Two calling conventions
   - Special register allocation rules
   - Manual resource management (X30)
   - All for what is essentially "a function that captures some variables"

   Maybe a simpler approach (all functions are closures, even if empty) would be better for a prototype?

2. **What's the actual performance difference?** We made closures work, but are they actually fast enough? We haven't benchmarked or compared to a real Clojure implementation.

3. **How would this scale?** What happens when we add:
   - Multiple arities?
   - Variadic functions?
   - Recursive closures?
   - Mutual recursion?
   - Would the current design handle these cleanly?

4. **Is the register allocator robust enough?** We found a subtle bug with v8 conflicts. Are there other similar bugs waiting to happen? Should we have better validation/assertions?

### If I Could Redesign From Scratch

I might consider:

1. **Separate temp vs argument register pools explicitly** - as shown above with the enum design

2. **Unified function representation (everything is a closure)**
   - Simpler codegen (one calling convention)
   - Optimize later with escape analysis
   - Prototype first, optimize second

3. **Automatic external call wrapping** - as shown above

4. **Constant-based offset calculations** - define heap layout as constants, compute offsets from structure

But... we'd be starting over, and what we have **works**. It's not perfect, but it's functional, and we learned a lot building it.

### Bottom Line

The current design is a **working compromise**. It has warts (the register numbering hack, the manual X30 preservation, the magic numbers), but it achieves the goal of making closures work efficiently.

For a research prototype or learning exercise, this is probably fine. For production code, I'd want to clean up the rough edges and add more robust testing.

**My honest assessment**: I'm proud we got it working, but I'm not entirely happy with how we got there. Some of the fixes feel like duct tape rather than proper solutions. But sometimes that's how systems are built - you solve the problem in front of you, learn from it, and refactor later if needed.

The implementation demonstrates that closures CAN work in a JIT compiler, and the performance should be reasonable. The architecture (tagged dispatch, inline checks, pre-allocated registers) is sound. The execution is... functional, if imperfect.

If this were a real project, I'd advocate for:
- Refactoring the virtual register system to be more explicit
- Adding constants for all magic numbers
- Automating X30 preservation
- Adding comprehensive tests for edge cases
- Benchmarking against other implementations

But for what it is - a proof of concept that closures work in our JIT - it succeeds.
