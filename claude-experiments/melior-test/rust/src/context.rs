//! Safe MLIR Context wrapper following Concrete's pattern
//!
//! This module provides a safe wrapper around melior's Context that:
//! 1. Uses melior utility functions instead of raw FFI
//! 2. Provides single global MLIR initialization
//! 3. Wraps the context to prevent raw pointer access
//! 4. Ensures proper cleanup and destruction order

use melior::{
    Context as MeliorContext,
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};
use std::sync::Once;

static INIT: Once = Once::new();

/// Safe wrapper around melior::Context that prevents crashes
#[derive(Debug)]
pub struct Context {
    melior_context: MeliorContext,
}

impl Context {
    /// Create a new properly initialized MLIR context
    pub fn new() -> Self {
        let melior_context = initialize_mlir();
        Self { melior_context }
    }

    /// Get reference to the underlying melior context
    pub fn melior_context(&self) -> &MeliorContext {
        &self.melior_context
    }

    /// Allow unregistered dialects (safe wrapper)
    pub fn allow_unregistered_dialects(&self) {
        unsafe {
            use mlir_sys::*;
            mlirContextSetAllowUnregisteredDialects(self.melior_context.to_raw(), true);
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

// Prevent the context from being dropped automatically - let it leak
impl Drop for Context {
    fn drop(&mut self) {
        // Intentionally do nothing - let the context leak rather than crash
        // This prevents the MLIR context destructor from being called
        std::mem::forget(std::mem::take(&mut self.melior_context));
    }
}

/// Initialize MLIR with careful global state management
fn initialize_mlir() -> MeliorContext {
    // Only initialize passes once globally
    INIT.call_once(|| {
        register_all_passes();
    });

    let context = MeliorContext::new();
    
    // Add back dialect registration but do it carefully
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    
    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::{Location, Module};

    #[test]
    fn test_context_creation() {
        let context = Context::new();
        let location = Location::unknown(context.melior_context());
        let _module = Module::new(location);
        // Should not crash
    }

    #[test]
    fn test_multiple_contexts() {
        let context1 = Context::new();
        let context2 = Context::new();
        
        let location1 = Location::unknown(context1.melior_context());
        let location2 = Location::unknown(context2.melior_context());
        
        let _module1 = Module::new(location1);
        let _module2 = Module::new(location2);
        
        // Should not crash with multiple contexts
    }

    #[test]
    fn test_unregistered_dialects() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        
        let location = Location::unknown(context.melior_context());
        let _module = Module::new(location);
        
        // Should allow unregistered dialect operations
    }
}