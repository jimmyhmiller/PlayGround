use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module},
    utility::register_all_dialects,
};

/// Wrapper around MLIR context with convenient initialization
pub struct MlirContext {
    context: Context,
}

impl MlirContext {
    /// Create a new MLIR context with all dialects registered
    pub fn new() -> Self {
        let context = Context::new();
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        Self { context }
    }

    /// Get the underlying context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Create a new module with unknown location
    pub fn create_module(&self) -> Module<'_> {
        Module::new(Location::unknown(&self.context))
    }

    /// Create a location (for now, always unknown)
    pub fn location(&self) -> Location<'_> {
        Location::unknown(&self.context)
    }
}

impl Default for MlirContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = MlirContext::new();
        let _module = ctx.create_module();
        // If we get here without panicking, success!
    }
}
