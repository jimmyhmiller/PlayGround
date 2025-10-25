# Integration Tests

This directory contains integration tests for the MLIR-Lisp REPL.

## Files

### Test Infrastructure

- **`repl_test.zig`** - Zig integration test suite
  - Spawns REPL process, pipes input, captures output
  - Integrated into `zig build test`
  - Run with: `zig build test`

- **`test_repl_runner.sh`** - Shell script test runner
  - Quick executable test suite (6 tests)
  - Run from project root: `./test/integration/test_repl_runner.sh`
  - All tests should pass ✅

### Test Input Files

Test files used by the integration tests:

- `test_repl_autoexec.txt` - Simple constant auto-execution test
- `test_repl_complete.txt` - Complex multi-operation test
- `test_repl_simple_workflow.txt` - Multiple operation accumulation test
- `test_func_call.txt` - Function call test (basic)
- `test_func_call2.txt` - Function call test (variant)
- `test_func_call_correct.txt` - Function call test (corrected syntax)
- `test_call_module.mlir-lisp` - Complete module with function call
- `test_just_call.mlir-lisp` - Wrapper function test
- `test_call.mlir` - MLIR syntax reference

## Running Tests

### Quick Test (Shell Script)
```bash
# From project root
./test/integration/test_repl_runner.sh
```

### Full Test Suite (Zig)
```bash
# From project root
zig build test
```

## Test Coverage

Current tests verify:
- ✅ REPL help command
- ✅ Simple constant auto-execution
- ✅ Function definitions (multi-line)
- ✅ :mlir command (module display)
- ✅ :clear command
- ✅ Unbalanced bracket detection

All 6 tests passing as of last run.
