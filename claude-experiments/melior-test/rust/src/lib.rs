// pub mod context; // Temporarily disabled due to mlir_sys issue
pub mod tensor_ops_dialect;
// pub mod tensor_ops_ffi; // Temporarily disabled due to mlir_sys issue
// pub mod transform_pdl; // Temporarily disabled due to RegionLike issue
// pub mod custom_dialect; // Temporarily disabled for compilation
// pub mod dialect_transforms; // Temporarily disabled for compilation 
// pub mod jit_engine; // Temporarily disabled for compilation
// pub mod tensor_pdl_patterns; // Temporarily disabled due to compilation issues
// pub mod transform_matching; // Temporarily disabled due to compilation issues
// pub mod tensor_ops_lowering; // Temporarily disabled for context migration
// TODO: Re-enable when C++ dialect is built
// pub mod tensor_ops_proper;

// pub use context::Context; // Temporarily disabled
pub use tensor_ops_dialect::TensorOpsDialect;
// pub use transform_pdl::{TransformDialect, PdlDialect, TransformPdlBuilder}; // Temporarily disabled
// pub use custom_dialect::CalcDialect; // Temporarily disabled
// pub use dialect_transforms::{CalcToStandardTransform, LLVMLowering}; // Temporarily disabled
// pub use jit_engine::{JitEngine, CompilationPipeline, BenchmarkUtils, BenchmarkResults}; // Temporarily disabled
// pub use tensor_pdl_patterns::{TensorPdlPatterns, TensorPatternCollection}; // Temporarily disabled
// pub use transform_matching::{TensorTransformOps, TensorMatcher, TensorTransformPipeline}; // Temporarily disabled
// pub use tensor_ops_lowering::TensorOpsLowering; // Temporarily disabled
// TODO: Re-enable when C++ dialect is built
// pub use tensor_ops_proper::TensorOpsDialect as ProperTensorOpsDialect;
