# MLIR-Lisp

A low-level, statically typed Lisp that compiles to MLIR. The language provides one-to-one mapping with MLIR operations, with higher-level constructs built using macros.

## Features

- **Direct MLIR Integration**: All MLIR operations are available as special forms
- **JIT Compilation**: Execute code directly via MLIR's JIT engine
- **Interactive REPL**: Experiment with MLIR operations interactively
- **S-expression Syntax**: Lisp-style syntax for MLIR operations
- **Type Safety**: Statically typed with full MLIR type support

## Building

Requirements:
- Zig 0.15.1
- LLVM/MLIR 20.1.7

```bash
zig build
```

## Usage

### Run a file
```bash
./zig-out/bin/mlir_lisp [--generic|-g] examples/fibonacci.mlir-lisp
```

Options:
- `--generic` or `-g`: Print MLIR in generic form (shows all attributes)

### Interactive REPL
```bash
./zig-out/bin/mlir_lisp --repl
```

## Examples

See the `examples/` directory for sample programs:
- `fibonacci.mlir-lisp` - Recursive fibonacci with main function
- `add.lisp` - Basic arithmetic operations

## Language Syntax

See [`docs/grammar.md`](docs/grammar.md) for detailed syntax documentation.

### Quick Example

```lisp
(mlir
  (operation
    (name func.func)
    (attributes {:sym_name @add :function_type (!function (inputs i32 i32) (results i32))})
    (regions
      (region
        (block
          (arguments [[%a i32] [%b i32]])
          (operation
            (name arith.addi)
            (result-bindings [%result])
            (result-types i32)
            (operand-uses %a %b))
          (operation
            (name func.return)
            (operand-uses %result)))))))
```

## Documentation

- [`docs/grammar.md`](docs/grammar.md) - Language syntax and grammar
- [`docs/mlir_integration.md`](docs/mlir_integration.md) - MLIR integration details
- [`docs/reader.md`](docs/reader.md) - S-expression reader implementation
- [`claude.md`](claude.md) - Project instructions and tool documentation

## Development

### Testing
```bash
zig build test
```

### Status Files
- `BUGS.md` - Tracked bugs (managed by bug-tracker tool)
- `REPL_TEST_STATUS.md` - REPL test coverage status

## License

[Add your license here]
