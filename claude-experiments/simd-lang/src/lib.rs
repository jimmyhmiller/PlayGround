//! simd-lang as a library: the pure-Rust SIMD JavaScript lexer.
//!
//! - [`stage1`] — the SIMD token-boundary classifier (NEON on aarch64, scalar
//!   elsewhere). Produces the two bitmaps the lexer consumes, byte-identical to
//!   the MLIR `examples/js_stage1.simd` contract.
//! - [`js`] — the stage-2 pull `Lexer` / `tokenize` (the parser↔lexer protocol).
//!
//! Depend on this with `default-features = false` to get just the lexer (no
//! MLIR/melior). The `.simd` compiler + JIT live behind the `mlir` feature and
//! the binary target.

pub mod js;
pub mod stage1;
