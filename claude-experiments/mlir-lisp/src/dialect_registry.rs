/// Dialect Registry
///
/// Maintains a registry of IRDL dialect definitions and Transform patterns.
/// This enables runtime loading of dialects and transformations.

use crate::parser::Value;
use std::collections::HashMap;

/// Represents an IRDL operation definition
#[derive(Debug, Clone)]
pub struct IrdlOperation {
    pub name: String,
    pub summary: String,
    pub description: String,
    pub operands: Vec<Value>,
    pub results: Vec<Value>,
    pub attributes: Vec<Value>,
    pub traits: Vec<Value>,
    pub constraints: Vec<Value>,
}

/// Represents an IRDL dialect definition
#[derive(Debug, Clone)]
pub struct IrdlDialect {
    pub name: String,
    pub namespace: String,
    pub description: String,
    pub operations: Vec<IrdlOperation>,
}

/// Represents a Transform definition
#[derive(Debug, Clone)]
pub struct TransformDefinition {
    pub name: String,
    pub description: String,
    pub body: Value,
}

/// Represents a PDL Pattern definition
#[derive(Debug, Clone)]
pub struct PdlPattern {
    pub name: String,
    pub benefit: i64,
    pub description: String,
    pub match_body: Value,
    pub rewrite_body: Value,
    pub constraints: Vec<Value>,
}

/// Registry for dialects and transformations
pub struct DialectRegistry {
    dialects: HashMap<String, IrdlDialect>,
    transforms: HashMap<String, TransformDefinition>,
    patterns: HashMap<String, PdlPattern>,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self {
            dialects: HashMap::new(),
            transforms: HashMap::new(),
            patterns: HashMap::new(),
        }
    }

    /// Register a dialect from an expanded form
    /// Expected form: (irdl-dialect-definition name namespace description ops)
    pub fn register_dialect(&mut self, expanded: &Value) -> Result<(), String> {
        if let Value::List(elements) = expanded {
            if elements.len() < 5 {
                return Err("Invalid irdl-dialect-definition form".into());
            }

            // Extract name, namespace, description
            let name = match &elements[1] {
                Value::String(s) => s.clone(),
                _ => return Err("Dialect name must be a string".into()),
            };

            let namespace = match &elements[2] {
                Value::String(s) => s.clone(),
                _ => return Err("Dialect namespace must be a string".into()),
            };

            let description = match &elements[3] {
                Value::String(s) => s.clone(),
                _ => return Err("Dialect description must be a string".into()),
            };

            // Extract operations
            let ops_vec = match &elements[4] {
                Value::Vector(v) => v,
                _ => return Err("Dialect operations must be a vector".into()),
            };

            let mut operations = Vec::new();
            for op in ops_vec {
                let parsed_op = self.parse_operation(op)?;
                operations.push(parsed_op);
            }

            let dialect = IrdlDialect {
                name: name.clone(),
                namespace,
                description,
                operations,
            };

            self.dialects.insert(name, dialect);
            Ok(())
        } else {
            Err("Expected list for dialect definition".into())
        }
    }

    /// Parse an operation from expanded form
    fn parse_operation(&self, op_form: &Value) -> Result<IrdlOperation, String> {
        if let Value::List(elements) = op_form {
            if elements.len() < 3 {
                return Err("Invalid irdl-op-definition form".into());
            }

            // Skip the "irdl-op-definition" symbol
            let name = match &elements[1] {
                Value::String(s) => s.clone(),
                _ => return Err("Operation name must be a string".into()),
            };

            let metadata = match &elements[2] {
                Value::Map(m) => m,
                _ => return Err("Operation metadata must be a map".into()),
            };

            // Extract metadata fields
            let mut summary = String::new();
            let mut description = String::new();
            let mut operands = Vec::new();
            let mut results = Vec::new();
            let mut attributes = Vec::new();
            let mut traits = Vec::new();
            let mut constraints = Vec::new();

            for (key, value) in metadata {
                match key {
                    Value::Keyword(k) => match k.as_str() {
                        "summary" => {
                            if let Value::String(s) = value {
                                summary = s.clone();
                            }
                        }
                        "description" => {
                            if let Value::String(s) = value {
                                description = s.clone();
                            }
                        }
                        "operands" => {
                            if let Value::Vector(v) = value {
                                operands = v.clone();
                            }
                        }
                        "results" => {
                            if let Value::Vector(v) = value {
                                results = v.clone();
                            }
                        }
                        "attributes" => {
                            if let Value::Vector(v) = value {
                                attributes = v.clone();
                            }
                        }
                        "traits" => {
                            if let Value::Vector(v) = value {
                                traits = v.clone();
                            }
                        }
                        "constraints" => {
                            if let Value::Vector(v) = value {
                                constraints = v.clone();
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            Ok(IrdlOperation {
                name,
                summary,
                description,
                operands,
                results,
                attributes,
                traits,
                constraints,
            })
        } else {
            Err("Expected list for operation definition".into())
        }
    }

    /// Register a transform from an expanded form
    /// Expected form: (transform-definition name description body)
    pub fn register_transform(&mut self, expanded: &Value) -> Result<(), String> {
        if let Value::List(elements) = expanded {
            if elements.len() < 4 {
                return Err("Invalid transform-definition form".into());
            }

            let name = match &elements[1] {
                Value::String(s) => s.clone(),
                _ => return Err("Transform name must be a string".into()),
            };

            let description = match &elements[2] {
                Value::String(s) => s.clone(),
                _ => return Err("Transform description must be a string".into()),
            };

            let body = elements[3].clone();

            let transform = TransformDefinition {
                name: name.clone(),
                description,
                body,
            };

            self.transforms.insert(name, transform);
            Ok(())
        } else {
            Err("Expected list for transform definition".into())
        }
    }

    /// Register a PDL pattern from an expanded form
    /// Expected form: (pdl-pattern-definition name {:benefit N :description "..." :match ... :rewrite ...})
    pub fn register_pattern(&mut self, expanded: &Value) -> Result<(), String> {
        if let Value::List(elements) = expanded {
            if elements.len() < 3 {
                return Err("Invalid pdl-pattern-definition form".into());
            }

            let name = match &elements[1] {
                Value::String(s) => s.clone(),
                _ => return Err("Pattern name must be a string".into()),
            };

            let metadata = match &elements[2] {
                Value::Map(m) => m,
                _ => return Err("Pattern metadata must be a map".into()),
            };

            let mut benefit = 1;
            let mut description = String::new();
            let mut match_body = Value::List(vec![]);
            let mut rewrite_body = Value::List(vec![]);
            let mut constraints = Vec::new();

            for (key, value) in metadata {
                match key {
                    Value::Keyword(k) => match k.as_str() {
                        "benefit" => {
                            if let Value::Integer(n) = value {
                                benefit = *n;
                            }
                        }
                        "description" => {
                            if let Value::String(s) = value {
                                description = s.clone();
                            }
                        }
                        "match" => {
                            match_body = value.clone();
                        }
                        "rewrite" => {
                            rewrite_body = value.clone();
                        }
                        "constraints" => {
                            if let Value::Vector(v) = value {
                                constraints = v.clone();
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            let pattern = PdlPattern {
                name: name.clone(),
                benefit,
                description,
                match_body,
                rewrite_body,
                constraints,
            };

            self.patterns.insert(name, pattern);
            Ok(())
        } else {
            Err("Expected list for pattern definition".into())
        }
    }

    /// Get a dialect by name
    pub fn get_dialect(&self, name: &str) -> Option<&IrdlDialect> {
        self.dialects.get(name)
    }

    /// Get a transform by name
    pub fn get_transform(&self, name: &str) -> Option<&TransformDefinition> {
        self.transforms.get(name)
    }

    /// Get a pattern by name
    pub fn get_pattern(&self, name: &str) -> Option<&PdlPattern> {
        self.patterns.get(name)
    }

    /// List all registered dialects
    pub fn list_dialects(&self) -> Vec<&str> {
        self.dialects.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered transforms
    pub fn list_transforms(&self) -> Vec<&str> {
        self.transforms.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered patterns
    pub fn list_patterns(&self) -> Vec<&str> {
        self.patterns.keys().map(|s| s.as_str()).collect()
    }

    /// Get all patterns as a vector
    pub fn get_all_patterns(&self) -> Vec<&PdlPattern> {
        self.patterns.values().collect()
    }

    /// Find an operation in a dialect
    pub fn find_operation(&self, dialect_name: &str, op_name: &str) -> Option<&IrdlOperation> {
        self.dialects.get(dialect_name)
            .and_then(|d| d.operations.iter().find(|op| op.name == op_name))
    }

    /// Process expanded forms and register them
    pub fn process_expanded_form(&mut self, expanded: &Value) -> Result<(), String> {
        if let Value::List(elements) = expanded {
            if let Some(Value::Symbol(s)) = elements.first() {
                match s.as_str() {
                    "irdl-dialect-definition" => {
                        self.register_dialect(expanded)?;
                    }
                    "transform-definition" => {
                        self.register_transform(expanded)?;
                    }
                    "pdl-pattern-definition" => {
                        self.register_pattern(expanded)?;
                    }
                    _ => {
                        // Not a registry form, skip
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_registry() {
        let mut registry = DialectRegistry::new();

        // Create a simple dialect definition
        let dialect = Value::List(vec![
            Value::Symbol("irdl-dialect-definition".to_string()),
            Value::String("test".to_string()),
            Value::String("test".to_string()),
            Value::String("Test dialect".to_string()),
            Value::Vector(vec![]),
        ]);

        registry.register_dialect(&dialect).unwrap();

        assert!(registry.get_dialect("test").is_some());
        assert_eq!(registry.list_dialects().len(), 1);
    }
}
