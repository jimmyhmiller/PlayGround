use melior::ir::r#type::Type;
use std::collections::HashMap;

/// Function signature for forward declarations
#[derive(Clone)]
pub struct FunctionSignature<'c> {
    pub arg_types: Vec<Type<'c>>,
    pub return_type: Type<'c>,
}

/// Registry of function signatures for two-pass compilation
/// Allows functions to call each other (including recursively) regardless of definition order
pub struct FunctionRegistry<'c> {
    functions: HashMap<String, FunctionSignature<'c>>,
}

impl<'c> FunctionRegistry<'c> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a function signature
    pub fn register(&mut self, name: String, arg_types: Vec<Type<'c>>, return_type: Type<'c>) {
        self.functions.insert(name, FunctionSignature {
            arg_types,
            return_type,
        });
    }

    /// Check if a function is registered
    pub fn is_declared(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get a function signature
    pub fn get(&self, name: &str) -> Option<&FunctionSignature<'c>> {
        self.functions.get(name)
    }

    /// Get return type for a function
    pub fn get_return_type(&self, name: &str) -> Option<Type<'c>> {
        self.functions.get(name).map(|sig| sig.return_type)
    }
}
