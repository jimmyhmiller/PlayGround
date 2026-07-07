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
//! `runtime.rs` ties them: reader (code is data), `macroexpand` (re-enters
//! compiled code mid-compile), `analyze`, and value-model-aware primitives.
//!
//! The three micro-languages in `examples/` exercise it:
//!   * `calc_fixnum` / `calc_float` — same generic engine, two value models.
//!   * `microlisp` — macros, `defmacro`, incremental eval on the fixnum model.

pub mod bigint;
pub mod bytecode;
pub mod cek;
pub mod code;
pub mod compiled;
pub mod dispatch;
pub mod gc;
pub mod ir;
pub mod model;
pub mod runtime;
pub mod speculation;
pub mod value;

pub use bytecode::{BytecodeVm, ModelEmit};
pub use cek::CekMachine;
pub use code::{CodeSpace, Traced, TreeWalk};
pub use compiled::ClosureComp;
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
