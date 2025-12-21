use melior::{
    ir::{
        r#type::{FunctionType, IntegerType},
        Location, Module, Type,
    },
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};

/// Dialect registry for managing MLIR dialects
/// With Melior, this is much simpler than the Zig version since Melior handles
/// dialect registration automatically.
pub struct DialectRegistry {
    context: Context,
}

impl DialectRegistry {
    pub fn new() -> Self {
        let context = Context::new();

        // Register all dialects and LLVM translations
        let registry = melior::dialect::DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        Self { context }
    }

    /// Get the MLIR context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Create a new empty module
    pub fn create_module(&self) -> Module<'_> {
        let location = Location::unknown(&self.context);
        Module::new(location)
    }

    /// Parse a type string into an MLIR type
    pub fn parse_type(&self, type_str: &str) -> Option<Type<'_>> {
        Type::parse(&self.context, type_str)
    }

    /// Get common integer types
    pub fn i1_type(&self) -> Type<'_> {
        IntegerType::new(&self.context, 1).into()
    }

    pub fn i8_type(&self) -> Type<'_> {
        IntegerType::new(&self.context, 8).into()
    }

    pub fn i16_type(&self) -> Type<'_> {
        IntegerType::new(&self.context, 16).into()
    }

    pub fn i32_type(&self) -> Type<'_> {
        IntegerType::new(&self.context, 32).into()
    }

    pub fn i64_type(&self) -> Type<'_> {
        IntegerType::new(&self.context, 64).into()
    }

    /// Get common float types
    pub fn f16_type(&self) -> Type<'_> {
        Type::float16(&self.context)
    }

    pub fn f32_type(&self) -> Type<'_> {
        Type::float32(&self.context)
    }

    pub fn f64_type(&self) -> Type<'_> {
        Type::float64(&self.context)
    }

    /// Get index type
    pub fn index_type(&self) -> Type<'_> {
        Type::index(&self.context)
    }

    /// Get function type
    pub fn function_type<'a>(&'a self, inputs: &[Type<'a>], results: &[Type<'a>]) -> Type<'a> {
        FunctionType::new(&self.context, inputs, results).into()
    }
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_registry_creation() {
        let registry = DialectRegistry::new();
        let _module = registry.create_module();
    }

    #[test]
    fn test_type_parsing() {
        let registry = DialectRegistry::new();

        // Test basic integer types
        let i32_type = registry.parse_type("i32");
        assert!(i32_type.is_some());

        let i64_type = registry.parse_type("i64");
        assert!(i64_type.is_some());

        // Test float types
        let f32_type = registry.parse_type("f32");
        assert!(f32_type.is_some());

        // Test index type
        let index_type = registry.parse_type("index");
        assert!(index_type.is_some());
    }

    #[test]
    fn test_type_helpers() {
        let registry = DialectRegistry::new();

        let _i32 = registry.i32_type();
        let _i64 = registry.i64_type();
        let _f32 = registry.f32_type();
        let _f64 = registry.f64_type();
        let _index = registry.index_type();
    }

    #[test]
    fn test_function_type() {
        let registry = DialectRegistry::new();

        let i32_type = registry.i32_type();
        let i64_type = registry.i64_type();

        let _func_type = registry.function_type(&[i32_type, i32_type], &[i64_type]);
    }
}
