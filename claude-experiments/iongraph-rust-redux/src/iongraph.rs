// Legacy module - re-exports from compilers::ion for backward compatibility
// New code should use compilers::ion::schema directly

pub use crate::compilers::ion::schema::*;
pub use crate::compilers::ion::migration::migrate;
