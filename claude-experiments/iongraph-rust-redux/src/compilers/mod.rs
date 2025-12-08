pub mod ion;
pub mod universal;
pub mod dot;

pub use ion::ir_impl::IonIR;
pub use universal::ir_impl::UniversalCompilerIR;
pub use dot::{parse_dot, dot_to_universal, DotGraph};
