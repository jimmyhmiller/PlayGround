//! microlang — a trait sketch for a dynamic-language toolkit, re-cut at the two
//! axes the original toolkit conflated:
//!
//!   * the **value axis** (`model.rs`): `Repr` carved at *immediacy*, plus
//!     `ValueModel`. Two real impls (`LowBitModel`, `NanBoxModel`) drive the
//!     SAME generic arithmetic and differ only in which numeric category is
//!     immediate.
//!
//!   * the **execution axis** (`code.rs`): `CodeSpace` decouples meaning (`Ir`)
//!     from strategy. `TreeWalk` is the interpreter tier; a JIT would be a
//!     second impl over the same `Ir` and the same re-entrant `invoke`.
//!
//! `runtime.rs` ties them: reader (code is data), `analyze` (`Val` -> `Ir`), and
//! value-model-aware primitives. The toolkit has NO macro system of its own — a
//! frontend that wants macros expands the `Val` tree before `analyze` (the
//! `mclj` frontend does exactly this, procedurally).

pub mod bigint;
pub mod bytecode;
pub mod cek;
pub mod code;
pub mod compiled;
pub mod dispatch;
pub mod gc;
pub mod ir;
#[cfg(feature = "jit")]
pub mod jit_cranelift;
pub mod model;
pub mod optimize;
pub mod runtime;
/// An OPTIONAL s-expression frontend (reader + a little-Lisp `Val`->`Ir`
/// compiler). NOT part of the core axes — a convenience a frontend may use.
pub mod sexpr;
pub mod speculation;
pub mod value;

pub use bytecode::{BytecodeVm, ModelEmit};
pub use cek::CekMachine;
pub use code::{CodeSpace, Traced, TreeWalk};
pub use optimize::Optimized;
pub use compiled::ClosureComp;
#[cfg(feature = "jit")]
pub use jit_cranelift::{jit_can_compile, JitCranelift, ModelArithJit, Tiered};
pub use dispatch::{Dispatch, DispatchStats, Megamorphic, MonomorphicIc, PolymorphicIc};
pub use speculation::{
    AlwaysMonomorphic, BlacklistAfter, Decision, NeverSpeculate, SpecCounters, SpecStats,
    Speculative, SpeculationPolicy,
};
pub use model::{
    HighBit, HighBitModel, LowBit, LowBitModel, NanBox, NanBoxModel, Repr, ValueModel,
};
pub use runtime::Runtime;
pub use value::{Cat, Obj, Val};
