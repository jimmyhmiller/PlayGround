pub mod types;
pub mod ir;
pub mod builder;
pub mod verify;
pub mod display;
pub mod interp;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod expressiveness_tests;
#[cfg(test)]
mod interp_tests;

pub use types::{Type, Signature};
pub use ir::{Value, BlockId, FuncRef, CmpOp, OverflowOp, DeoptId, DeoptInfo, Inst, Terminator, InstNode, Block, ExternFunc, Function};
pub use builder::FunctionBuilder;
pub use verify::{verify, VerifyError};
pub use interp::{Interpreter, InterpResult, InterpError, ExternCallResult};
