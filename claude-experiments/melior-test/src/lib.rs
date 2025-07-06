pub mod tensor_ops_dialect;
pub mod tensor_ops_lowering;

pub use tensor_ops_dialect::TensorOpsDialect;
pub use tensor_ops_lowering::{TensorOpsLowering, TensorOpsPassManager};