//! IR-to-IR transform passes over the JSIR IR.
//!
//! First pass: **dead code elimination** (`dce`). Upstream's DCE
//! (`maldoca/js/ir/transforms/dead_code_elimination`) only eliminates `if`/
//! `while` whose test is a *syntactic* boolean literal, plus unused symbols.
//! Ours is driven by the constant-propagation **dataflow analysis**, so it also
//! folds computed-constant conditions (`if (2 > 1)`, `if (x)` where `x` is a
//! propagated constant), and it removes statically-unreachable code after a
//! terminator. It is intentionally *sound*: it never removes anything that
//! could have a side effect or change observable behavior.

pub mod dce;
pub mod memoize;
pub mod treeshake;

pub use dce::{eliminate_dead_code, eliminate_dead_code_with_roots, Stats};
pub use treeshake::{tree_shake, TreeShakeResult, TreeShakeStats};
