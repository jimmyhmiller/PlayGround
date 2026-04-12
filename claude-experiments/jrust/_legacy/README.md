# Legacy Java Bootstrap Compiler

This directory contains the original Java implementation of the JRust compiler that was used to bootstrap the self-hosting compiler.

It is no longer needed — the self-hosting compiler (`compiler.jrs`) is now bootstrapped from the checked-in stage0 binaries in `stages/stage0/`.

Contents:
- `src/` — Java source for the bootstrap compiler
- `build/` — Compiled Java classes
- `build.sh` — Script to compile the Java source
