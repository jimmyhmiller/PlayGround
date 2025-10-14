# Bugs

This file tracks bugs discovered during development.

## Pointer arithmetic not working [pesky-tidy-dingo]

**ID:** pesky-tidy-dingo
**Timestamp:** 2025-10-14 17:06:20
**Severity:** high
**Location:** src/type_checker.zig
**Tags:** type-system, pointers, arithmetic

### Description

Pointer arithmetic (e.g., adding integers to pointers) is not currently supported. Need to implement pointer arithmetic to work exactly like C: pointer + integer should move the pointer by that many elements.

---

