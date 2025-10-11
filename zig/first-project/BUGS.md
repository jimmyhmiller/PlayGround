# Bugs

This file tracks bugs discovered during development.

## Recursive functions with heap arrays and pointer indexing [excited-staid-sawfish]

**ID:** excited-staid-sawfish
**Timestamp:** 2025-10-11 00:31:29
**Severity:** medium
**Location:** tests/integration/50_complex_recursion.lisp
**Tags:** recursion, heap-arrays, pointer-index-read, functions

### Description

Test 50_complex_recursion.lisp implements recursive sum_array_recursive function using pointer-index-read on heap-allocated array. Combination of recursion, heap arrays, and pointer indexing fails. Expected sum: 15 (1+2+3+4+5).

---

