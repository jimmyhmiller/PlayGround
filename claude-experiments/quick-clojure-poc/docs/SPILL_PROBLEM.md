# Register Spilling Problem in Deeply Nested Expressions

## Problem Summary

When compiling deeply nested expressions like `(+ 1 (+ 1 (+ 1 ...)))` with 200 levels of nesting:

- **Our implementation**: 193 spill slots
- **Beagle**: 1-2 spill slots per function

This 100x difference causes our implementation to crash with SIGSEGV when stack offsets exceed ARM64's addressing limits.

## Root Cause

Our polymorphic arithmetic generates ~15+ virtual registers per `+` operation:

```
; For each (+ a b):
GetTag(tag_a, a)           ; Check if a is int
GetTag(tag_b, b)           ; Check if b is int
JumpIf(both_int, ...)      ; Branch on type
Untag(untagged_a, a)       ; Remove tag from a
Untag(untagged_b, b)       ; Remove tag from b
AddInt(result, a, b)       ; Integer add
Tag(tagged_result, result) ; Re-tag result
; ... plus float conversion paths with more registers
```

With 200 nested additions, we create ~3000 virtual registers. Since they're generated depth-first (innermost expression first), many registers remain live across the entire computation.

## Why Beagle Doesn't Have This Problem

Beagle uses a different compilation strategy:

1. **Simpler type system**: Beagle knows types at compile time, avoiding runtime tag checks
2. **Stack-based evaluation**: Results are consumed immediately, not held in registers
3. **Different IR structure**: Beagle's IR allows register reuse more aggressively

## Solutions to Investigate

1. **Type inference**: If we know both operands are integers, skip the polymorphic dispatch
2. **Eager evaluation**: Evaluate subexpressions and store results before moving to siblings
3. **Register pressure aware compilation**: Restructure IR to minimize live ranges
4. **Spill slot reuse**: Reuse spill slots when their values are dead

## Test Case

```clojure
; This should return 200, currently crashes
(+ 1 (+ 1 (+ 1 ... ))) ; 200 levels deep
```

## Comparison Data

| Depth | Our Spills | Beagle Spills | Our Result | Beagle Result |
|-------|------------|---------------|------------|---------------|
| 7     | 0          | 0             | 7          | 7             |
| 50    | ~48        | 1-2           | crashes    | 50            |
| 200   | 193        | 1-2           | crashes    | 200           |
