pub mod context;
pub mod tensor_ops_dialect;
pub mod tensor_ops_ffi;
pub mod tensor_ops_lowering;
// TODO: Re-enable when C++ dialect is built
// pub mod tensor_ops_proper;

pub use context::Context;
pub use tensor_ops_dialect::TensorOpsDialect;
pub use tensor_ops_lowering::TensorOpsLowering;
// TODO: Re-enable when C++ dialect is built
// pub use tensor_ops_proper::TensorOpsDialect as ProperTensorOpsDialect;
