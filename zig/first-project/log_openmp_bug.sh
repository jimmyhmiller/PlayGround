#!/bin/bash
cat >> BUGS.md << 'BUGEOF'

## playful-narwhal
**Title:** Test and implement OpenMP support for parallel compilation flags

**Description:** 
The compiler supports `(compiler-flag "-fopenmp")` syntax, but we haven't tested if OpenMP pragmas in generated C code actually compile and execute correctly. This would enable multi-threaded parallelization for compute-intensive operations.

**Context:**
- File: `src/simple_c_compiler.zig` (compiler flag handling)
- Current status: Untested whether OpenMP flags properly propagate to both compilation and linking
- Use case: Parallelizing matrix multiplication operations in GPT-2 implementation

**Reproduction:**
Try adding OpenMP pragma comments to generated C code and verify:
1. `(compiler-flag "-fopenmp")` passes flag to both compile and link stages
2. Generated C code with `#pragma omp parallel for` compiles without errors
3. Multi-threaded execution actually occurs at runtime
4. No segfaults or race conditions with generated code patterns

**Severity:** low (nice-to-have optimization)

**Tags:** performance, parallelization, openmp, compiler-flags

**Code snippet:**
```lisp
(compiler-flag "-fopenmp")
;; Would need to generate C code like:
;; #pragma omp parallel for
;; for (int i = 0; i < n; i++) { ... }
```

**Metadata:** {"related_issue": "matmul_optimization", "blocked_by": "none"}
BUGEOF
echo "Bug logged in BUGS.md as playful-narwhal"
