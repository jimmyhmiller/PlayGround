# Bugs

This file tracks bugs discovered during development.

## Pointer arithmetic not working [pesky-tidy-dingo] ✅ RESOLVED

**ID:** pesky-tidy-dingo
**Timestamp:** 2025-10-14 17:06:20
**Resolved:** 2025-10-14
**Severity:** high
**Location:** src/type_checker.zig
**Tags:** type-system, pointers, arithmetic

### Description

Pointer arithmetic (e.g., adding integers to pointers) is not currently supported. Need to implement pointer arithmetic to work exactly like C: pointer + integer should move the pointer by that many elements.

### Resolution

Added pointer arithmetic support in `synthesizeCBinaryOp` function in src/type_checker.zig:1560-1580. The implementation now correctly handles:
- `ptr + int` → returns pointer type
- `int + ptr` → returns pointer type
- `ptr - int` → returns pointer type
- `ptr - ptr` → returns `isize` (pointer difference)

---

