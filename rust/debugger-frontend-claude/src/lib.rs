//! Debugger Frontend Claude - A reusable library for custom language debugging
//! 
//! This library provides modular components for building debuggers on top of LLDB,
//! with a focus on custom language support including:
//! - Message protocol handling
//! - Memory and type analysis  
//! - Source-to-machine-code mapping
//! - Disassembly analysis
//! - Breakpoint management

pub mod core;
pub mod debugger;
pub mod analysis;
pub mod display;

// Re-export commonly used types
pub use core::*;
pub use debugger::*;
pub use analysis::*;
pub use display::*;

pub type Result<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_types() {
        let int_type = BuiltInTypes::Int;
        assert_eq!(int_type.get_tag(), 0b000);
        assert!(int_type.is_embedded());
    }
}
