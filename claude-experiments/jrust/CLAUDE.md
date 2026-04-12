# JRust Compiler

Self-hosting Rust-like language compiler targeting JVM bytecode. The compiler is written in JRust (`compiler.jrs`) and compiles itself.

## Bootstrap

**Always use `bash bootstrap.sh`** to update the compiler after making changes to `compiler.jrs`. Never manually copy files to `stages/stage0/`.

The bootstrap script is transactional — it verifies self-compilation, fixed point, and tests before modifying stage0. If any step fails, stage0 is untouched.

```bash
bash bootstrap.sh              # compile, verify, update stage0
bash bootstrap.sh --recover    # restore stage0 from git if broken
```

### Adding new syntax used by the compiler itself

This requires two bootstrap steps. The old stage0 can't parse syntax it doesn't know about yet.

1. Add the parser + codegen support for the new syntax, but **do not use it** in `compiler.jrs` yet
2. `bash bootstrap.sh` — gets the new parser into stage0
3. Now write code in `compiler.jrs` that uses the new syntax
4. `bash bootstrap.sh` — stage0 can now parse the new syntax

The bootstrap script detects parse failures and prints this reminder.

## Compiling and running programs

```bash
java -cp "stages/stage0:asm.jar" Main myfile.jrs    # compile
java -cp "output:asm.jar" Main                        # run
```

## Tests

```bash
bash run_tests.sh
```

Tests live in `tests/` — each test has a `.jrs` source file and a `.expected` output file.

## Project structure

- `compiler.jrs` — the self-hosting compiler (lexer, parser, type checker, codegen)
- `stages/stage0/` — compiled compiler classes (the bootstrap binary)
- `asm.jar` — ASM 9.8 bytecode library
- `bootstrap.sh` — safe transactional bootstrap script
- `run_tests.sh` — test runner
- `tests/` — test suite
- `LANGUAGE.md` — language reference
- `_legacy/` — original Java bootstrap compiler (not actively maintained)

## Current state

The type checker is implemented but disabled (commented out in `main()`). It correctly reads imported class signatures from `.class` files via ASM ClassReader. The remaining work is handling method name aliases — the codegen uses snake_case names (`visit_method`, `visit_insn`) for Java camelCase methods (`visitMethod`, `visitInsn`), and the type checker needs to understand these aliases.
