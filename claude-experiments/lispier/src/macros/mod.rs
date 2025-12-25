//! Macro system for Lispier
//!
//! Macros operate at the Value level, transforming s-expressions
//! before they are parsed into AST nodes.

pub mod builtins;
pub mod expander;
pub mod jit_macro;
pub mod registry;

pub use expander::MacroExpander;
pub use jit_macro::{JitMacro, JitMacroFn};
pub use registry::MacroRegistry;

use crate::value::Value;
use std::fmt;

/// Error type for macro expansion failures
#[derive(Debug, Clone)]
pub enum MacroError {
    /// Wrong number of arguments provided to macro
    WrongArity {
        macro_name: String,
        expected: String,
        got: usize,
    },
    /// Type mismatch in macro argument
    TypeError {
        macro_name: String,
        expected: &'static str,
        got: String,
    },
    /// General expansion failure
    ExpansionFailed(String),
    /// Maximum macro expansion depth exceeded (prevents infinite loops)
    MaxDepthExceeded,
    /// Invalid syntax in macro call
    InvalidSyntax {
        macro_name: String,
        message: String,
    },
    /// Dynamic macro compilation failed
    DynamicMacroCompilationFailed {
        macro_name: String,
        message: String,
    },
}

impl fmt::Display for MacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroError::WrongArity {
                macro_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "macro '{}': wrong number of arguments, expected {}, got {}",
                    macro_name, expected, got
                )
            }
            MacroError::TypeError {
                macro_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "macro '{}': expected {}, got {}",
                    macro_name, expected, got
                )
            }
            MacroError::ExpansionFailed(msg) => {
                write!(f, "macro expansion failed: {}", msg)
            }
            MacroError::MaxDepthExceeded => {
                write!(f, "maximum macro expansion depth exceeded")
            }
            MacroError::InvalidSyntax { macro_name, message } => {
                write!(f, "macro '{}': invalid syntax: {}", macro_name, message)
            }
            MacroError::DynamicMacroCompilationFailed { macro_name, message } => {
                write!(
                    f,
                    "failed to compile dynamic macro '{}': {}",
                    macro_name, message
                )
            }
        }
    }
}

impl std::error::Error for MacroError {}

/// Trait for implementing macros
///
/// A macro takes a slice of Value arguments (the form minus the macro name)
/// and returns an expanded Value.
pub trait Macro: Send + Sync {
    /// The name of the macro (used for lookup)
    fn name(&self) -> &str;

    /// Expand the macro given its arguments
    ///
    /// `args` is the list of arguments to the macro, excluding the macro name itself.
    /// For example, if the input is `(defn foo [x] x)`, args would be `[foo, [x], x]`.
    fn expand(&self, args: &[Value]) -> Result<Value, MacroError>;

    /// Optional documentation string for the macro
    fn doc(&self) -> Option<&str> {
        None
    }
}
