# Value Tagging and Equality Fix

## Problem

The JIT compiler had multiple issues with value representation:

1. **No distinction between nil, false, and 0** - All mapped to 0
2. **Equality returned numbers (0/1) instead of true/false**
3. **Equality comparisons were wrong** - `(= nil 0)` returned true because both untagged to 0

## Solution: Proper 3-bit Tagging (Following Beagle's Pattern)

### Tagging Scheme

Values now use 3-bit type tags in the low bits:

| Value | Binary Representation | Tag Bits | Decimal |
|-------|----------------------|----------|---------|
| **Integer n** | `(n << 3) \| 0b000` | `000` | n×8 |
| **Boolean** | `(val << 3) \| 0b011` | `011` | - |
| **Null** | `0b111` | `111` | 7 |

### Specific Values

| Value | Representation | Binary | Explanation |
|-------|---------------|--------|-------------|
| `nil` | 7 | `0b00111` | Special constant |
| `false` | 3 | `0b00011` | (0 << 3) \| 0b011 |
| `true` | 11 | `0b01011` | (1 << 3) \| 0b011 |
| `0` | 0 | `0b00000` | (0 << 3) \| 0b000 |
| `1` | 8 | `0b01000` | (1 << 3) \| 0b000 |
| `5` | 40 | `0b101000` | (5 << 3) \| 0b000 |

**Key Insight:** All values are now distinguishable by their tagged representation!

## Changes Made

### 1. Updated Value Constants (arm_codegen.rs:150-161)

```rust
IrValue::True => {
    // true: (1 << 3) | 0b011 = 11
    self.emit_mov_imm(dst_reg, 11);
}
IrValue::False => {
    // false: (0 << 3) | 0b011 = 3
    self.emit_mov_imm(dst_reg, 3);
}
IrValue::Null => {
    // nil: 0b111 = 7
    self.emit_mov_imm(dst_reg, 7);
}
```

### 2. Removed Result Untagging (arm_codegen.rs:117-120)

**Before:**
```rust
if result_physical != 0 {
    self.emit_mov(0, result_physical);
}
self.emit_asr_imm(0, 0, 3); // ← This was untagging the result!
```

**After:**
```rust
// Move result to x0 (keep it tagged)
if result_physical != 0 {
    self.emit_mov(0, result_physical);
}
```

### 3. Fixed Comparison to Return Tagged Booleans (arm_codegen.rs:462-468)

Comparison instructions now convert CSET's 0/1 result to tagged booleans:

```rust
// CSET sets dst to 1 if true, 0 if false
let instruction = 0x9A9F07E0 | (inverted_cond << 12) | (dst_reg as u32);
self.code.push(instruction);

// Convert 0/1 to tagged bools: 3 (false) or 11 (true)
self.emit_lsl_imm(dst_reg, dst_reg, 3);  // 0→0, 1→8
self.emit_add_imm(dst_reg, dst_reg, 3);  // 0→3, 8→11
```

### 4. Fixed Equality to Compare Tagged Values (compiler.rs:793-799)

**The Critical Fix:** Equality now compares tagged values directly without untagging.

**Before:**
```rust
let left_untagged = self.builder.new_register();
let right_untagged = self.builder.new_register();
self.builder.emit(Instruction::Untag(left_untagged, left));
self.builder.emit(Instruction::Untag(right_untagged, right));
self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::Equal));
```

This was wrong because:
- `nil` (7) → untag → 0
- `false` (3) → untag → 0
- Integer `0` (0) → untag → 0
All became 0, so they all compared equal!

**After:**
```rust
// Compare tagged values directly - this preserves type information
let result = self.builder.new_register();
self.builder.emit(Instruction::Compare(result, left, right, Condition::Equal));
```

Now:
- `nil` (7) vs `0` (0) → not equal ✓
- `false` (3) vs `0` (0) → not equal ✓
- `true` (11) vs `false` (3) → not equal ✓

### 5. Kept Numeric Comparisons Untagged (compiler.rs:755-762)

`<` and `>` still untag because they only work on integers:

```rust
fn compile_builtin_lt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
    // ...
    let left_untagged = self.builder.new_register();
    let right_untagged = self.builder.new_register();
    self.builder.emit(Instruction::Untag(left_untagged, left));
    self.builder.emit(Instruction::Untag(right_untagged, right));
    // Compare untagged integers
}
```

## Test Results

### Value Distinction
```clojure
nil    → 7
true   → 11
false  → 3
0      → 0
1      → 8
```

All values are distinct! ✓

### Equality Tests
```clojure
(= nil 0)       → false (3) ✓
(= nil false)   → false (3) ✓
(= false 0)     → false (3) ✓
(= true false)  → false (3) ✓
(= 5 5)         → true (11) ✓
(= 5 3)         → false (3) ✓
```

### Comparison Tests
```clojure
(< 1 2)  → true (11) ✓
(> 2 1)  → true (11) ✓
```

### Complex Example
```clojure
(let [x 2] (let [y 3] y) y)  → Compile error: Undefined variable: y ✓
(let [x 2])                   → nil (7) ✓
```

## Benefits

1. **Type Safety:** nil, false, and 0 are now properly distinct
2. **Correct Semantics:** Equality works as expected
3. **Proper Booleans:** Comparisons return true/false, not 1/0
4. **Clojure Compatibility:** Matches Clojure's behavior

## Notes

- Integer results are now tagged (e.g., `(+ 1 2)` returns 24 = 3×8)
- This is correct - values stay tagged throughout execution
- Only comparison operators produce booleans
- The tagging scheme follows Beagle's proven design
