pub mod builder;
pub mod display;
pub mod interp;
pub mod ir;
pub mod opt;
pub mod types;
pub mod verify;

pub use dynexec;

#[cfg(test)]
mod expressiveness_tests;
#[cfg(test)]
mod interp_tests;
#[cfg(test)]
mod tests;

pub use builder::{FunctionBuilder, ModuleBuilder};
pub use interp::{
    ConfiguredModuleInterpreter, ExternCallResult, InterpError, InterpResult, InterpRootManager,
    ModuleInterpreter, NoGcRoots,
};
pub use ir::{
    Block, BlockId, CmpOp, DeoptId, DeoptInfo, ExternFunc, FuncDef, FuncRef, Function, Inst,
    InstNode, Module, OverflowOp, PromptId, Terminator, Value,
};
pub use types::{Signature, Type};
pub use verify::{verify, verify_with, VerifyError, VerifyOptions};
