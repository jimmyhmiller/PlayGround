//! A principled, generic online partial evaluator.
//!
//! `engine` is the entire generic specializer; the other modules are clients
//! that reuse it verbatim (`bf`, `imp`, `fun`, `js`). `residual` is the shared
//! residual-program IR. This `lib` target exposes them so other crates (e.g. a
//! real-JavaScript frontend) can build on the engine and the `js` client.

pub mod bf;
pub mod engine;
pub mod fun;
pub mod imp;
pub mod js;
pub mod residual;
