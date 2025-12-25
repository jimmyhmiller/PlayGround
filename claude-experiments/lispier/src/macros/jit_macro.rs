//! JIT-compiled macro support
//!
//! Allows compiled functions to be registered and used as macros during expansion.

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Type alias for the JIT macro function signature
/// Takes a pointer to a Value::List containing the macro arguments
/// Returns a pointer to the expanded Value
pub type JitMacroFn = unsafe extern "C" fn(*const Value) -> *mut Value;

/// A macro backed by a JIT-compiled function
///
/// The function must have the signature `fn(*const Value) -> *mut Value`
/// where the input is the list of arguments and the output is the expanded form.
pub struct JitMacro {
    name: String,
    func_ptr: JitMacroFn,
}

impl JitMacro {
    /// Create a new JIT macro from a function pointer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `func_ptr` points to a valid function with the correct signature
    /// - The function is safe to call with any valid Value pointer
    pub unsafe fn new(name: impl Into<String>, func_ptr: JitMacroFn) -> Self {
        Self {
            name: name.into(),
            func_ptr,
        }
    }

    /// Get the raw function pointer
    pub fn func_ptr(&self) -> JitMacroFn {
        self.func_ptr
    }
}

impl Macro for JitMacro {
    fn name(&self) -> &str {
        &self.name
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // Create a Value::List from the arguments
        let args_list = Value::List(args.to_vec());

        // Call the JIT-compiled function
        let result_ptr = unsafe { (self.func_ptr)(&args_list as *const Value) };

        // Check for null pointer (error case)
        if result_ptr.is_null() {
            return Err(MacroError::ExpansionFailed(format!(
                "JIT macro '{}' returned null",
                self.name
            )));
        }

        // Take ownership of the returned Value
        let result = unsafe { Box::from_raw(result_ptr) };

        Ok(*result)
    }

    fn doc(&self) -> Option<&str> {
        None
    }
}

// JitMacro is Send + Sync because:
// - The function pointer is just an address (Copy)
// - The JIT-compiled code doesn't hold mutable state
// - We're careful about memory management in expand()
unsafe impl Send for JitMacro {}
unsafe impl Sync for JitMacro {}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple test macro that returns its input unchanged
    unsafe extern "C" fn identity_macro(input: *const Value) -> *mut Value {
        let value = unsafe { &*input };
        Box::into_raw(Box::new(value.clone()))
    }

    #[test]
    fn test_jit_macro_basic() {
        let jit_macro = unsafe { JitMacro::new("identity", identity_macro) };

        assert_eq!(jit_macro.name(), "identity");

        let args = vec![Value::Number(42.0)];
        let result = jit_macro.expand(&args).unwrap();

        // The result should be a list containing the number
        if let Value::List(items) = result {
            assert_eq!(items.len(), 1);
            assert!(matches!(items[0], Value::Number(n) if n == 42.0));
        } else {
            panic!("Expected list result");
        }
    }
}
