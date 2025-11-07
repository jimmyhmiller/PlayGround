# Bulk Test Failures - Missing Features

This document tracks the missing features discovered by running the parser against the entire Pyret codebase.

## Test Status

**Comparison Tests: 206 passed; 0 failed; 17 ignored**

## Files Copied for Testing

Three files from the Pyret codebase have been copied to `tests/pyret-files/` for full file comparison testing:

1. **let.arr** (8 lines) - From `tests/type-check/good/let.arr`
   - Features: let with multiple bindings and block syntax
   
2. **weave-tuple.arr** (23 lines) - From `tests/pyret/regression/weave-tuple.arr`
   - Features: tuple type annotations, tuple destructuring in function parameters
   
3. **option.arr** (33 lines) - From `src/arr/trove/option.arr`
   - Features: generic types, methods with trailing commas, real-world library code

## Missing Features (17 ignored tests)

### High Priority (Found in Real Pyret Code)

1. **Tuple Destructuring** (4 tests)
   - `{a; b} = {1; 2}` - Basic tuple destructuring in let bindings
   - `{shadow a; shadow b} = {1; 2; 3}` - With shadow keyword
   - Tuple destructuring in cases patterns
   - Needed for many stdlib files

2. **Provide-Types with Specific Types** (1 test)
   - `provide-types { Foo:: Foo, Bar:: Bar }`
   - Used in type-check test files
   
3. **Data Hiding in Provide** (1 test)
   - `provide: data Foo hiding(foo) end`
   - Used in matrices.arr and other stdlib files

4. **Underscore Partial Application** (2 tests)
   - `f = (_ + 2)` - Single underscore
   - `f = (_ + _)` - Multiple underscores
   - Common pattern in type-check tests

5. **Tuple Type Annotations** (1 test)
   - `f :: {Number; Number} -> {Number; Number}`
   - Used throughout type system tests

6. **Extract From Expression** (1 test)
   - `extract state from obj end`
   - Found in test-reactor.arr

7. **Dot Number Access** (1 test)
   - `t.0 + t.1 + t.2` - Shorthand for tuple access
   - Alternative to `t.{0}` syntax

### Medium Priority

8. **Provide From with Data** (1 test)
   - `provide from M: x, data Foo end`
   - Advanced module system feature

9. **Provide From Multiple Items** (1 test)
   - `provide from lists: map, filter end`
   - Already have basic `provide from` working

10. **Methods with Trailing Commas** (1 test)
    - Object methods ending with commas
    - Stylistic feature

11. **Spy with String** (1 test)
    - `spy "debug message": x end`
    - Debug feature enhancement

### Full File Tests (3 tests)

12. **test_full_file_let_arr** - Complete test file
13. **test_full_file_weave_tuple_arr** - Regression test file  
14. **test_full_file_option_arr** - Real stdlib file

## Invalid Tests Removed (4 tests)

These tests were removed after verification with the official Pyret parser:

1. ❌ `test_double_bang` - `list!!get(0)` is NOT valid Pyret syntax
2. ❌ `test_ellipsis_operator` - `[list: 1, 2, ...rest]` is NOT valid
3. ❌ `test_provide_types_with_braces` - `provide-types { Foo }` is NOT valid
4. ❌ `test_import_hiding` - `import lists as L hiding(map, filter)` is NOT valid

## Next Steps

To reach higher coverage of the Pyret codebase, implement features in this order:

1. **Tuple destructuring** - Blocks many files (highest impact)
2. **Underscore partial application** - Common in type system tests
3. **Data hiding** - Used in core libraries
4. **Tuple type annotations** - Needed for type system
5. **Provide-types with specific types** - For type system

## Files Currently Failing to Parse

See `bulk_test_results/failing_files.txt` for the complete list of ~173 files that fail to parse.

Common failure patterns:
- Tuple destructuring: ~30+ files
- Advanced provide/import: ~20+ files
- Type system features: ~40+ files
- Methods with commas: ~15+ files
- Extract from: ~5+ files

---
Last Updated: 2025-11-06
