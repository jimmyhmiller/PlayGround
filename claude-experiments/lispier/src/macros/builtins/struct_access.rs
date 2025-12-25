//! Struct field accessor macros
//!
//! These macros are dynamically created by defstruct to provide
//! field access that expands directly to LLVM operations.

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Macro for getting a struct field
/// Expands (Point/x p) to (llvm.load (llvm.getelementptr p [0, index]) : !llvm.ptr -> field_type)
pub struct StructFieldGetMacro {
    /// Full name of the macro (e.g., "Point/x")
    name: String,
    /// The struct type string (e.g., "!llvm.struct<(i64, i64)>")
    struct_type: String,
    /// Field index in the struct
    field_index: usize,
    /// Field type string (e.g., "i64")
    field_type: String,
}

impl StructFieldGetMacro {
    pub fn new(
        name: impl Into<String>,
        struct_type: impl Into<String>,
        field_index: usize,
        field_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            struct_type: struct_type.into(),
            field_index,
            field_type: field_type.into(),
        }
    }
}

impl Macro for StructFieldGetMacro {
    fn name(&self) -> &str {
        &self.name
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // Expect exactly one argument: the struct pointer
        if args.len() != 1 {
            return Err(MacroError::WrongArity {
                macro_name: self.name.clone(),
                expected: "1".to_string(),
                got: args.len(),
            });
        }

        let ptr = &args[0];

        // Build: (llvm.load (llvm.getelementptr inbounds p [0, field_index]) : !llvm.ptr -> field_type)
        //
        // The getelementptr needs:
        // - The pointer
        // - Indices: [0, field_index] - 0 to deref the pointer, field_index for the field
        // - Type annotation for the struct type
        //
        // Result:
        // (llvm.load
        //   {:elem_type field_type}
        //   (llvm.getelementptr {:elem_type struct_type :rawConstantIndices array<i32: 0, N>} ptr))

        // rawConstantIndices format: array<i32: 0, field_index>
        // The 0 dereferences the pointer, field_index selects the field
        let indices_str = format!("array<i32: 0, {}>", self.field_index);

        let gep = Value::List(vec![
            Value::symbol("llvm.getelementptr"),
            Value::Map(
                [
                    (
                        "elem_type".to_string(),
                        Value::symbol(&self.struct_type),
                    ),
                    (
                        "rawConstantIndices".to_string(),
                        Value::symbol(&indices_str),
                    ),
                    (
                        "result".to_string(),
                        Value::symbol("!llvm.ptr"),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            ptr.clone(),
        ]);

        let load = Value::List(vec![
            Value::symbol("llvm.load"),
            Value::Map(
                [
                    (
                        "elem_type".to_string(),
                        Value::symbol(&self.field_type),
                    ),
                    (
                        "result".to_string(),
                        Value::symbol(&self.field_type),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            gep,
        ]);

        Ok(load)
    }

    fn doc(&self) -> Option<&str> {
        None
    }
}

/// Macro for setting a struct field
/// Expands (Point/x! p v) to (llvm.store v (llvm.getelementptr p [0, index]))
pub struct StructFieldSetMacro {
    /// Full name of the macro (e.g., "Point/x!")
    name: String,
    /// The struct type string (e.g., "!llvm.struct<(i64, i64)>")
    struct_type: String,
    /// Field index in the struct
    field_index: usize,
    // Note: field_type not currently needed for set macro
    // but may be useful for future type checking
}

impl StructFieldSetMacro {
    pub fn new(
        name: impl Into<String>,
        struct_type: impl Into<String>,
        field_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            struct_type: struct_type.into(),
            field_index,
        }
    }
}

impl Macro for StructFieldSetMacro {
    fn name(&self) -> &str {
        &self.name
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // Expect exactly two arguments: the struct pointer and the value
        if args.len() != 2 {
            return Err(MacroError::WrongArity {
                macro_name: self.name.clone(),
                expected: "2".to_string(),
                got: args.len(),
            });
        }

        let ptr = &args[0];
        let value = &args[1];

        // Build: (llvm.store value (llvm.getelementptr inbounds p [0, field_index]))

        // rawConstantIndices format: array<i32: 0, field_index>
        let indices_str = format!("array<i32: 0, {}>", self.field_index);

        let gep = Value::List(vec![
            Value::symbol("llvm.getelementptr"),
            Value::Map(
                [
                    (
                        "elem_type".to_string(),
                        Value::symbol(&self.struct_type),
                    ),
                    (
                        "rawConstantIndices".to_string(),
                        Value::symbol(&indices_str),
                    ),
                    (
                        "result".to_string(),
                        Value::symbol("!llvm.ptr"),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            ptr.clone(),
        ]);

        let store = Value::List(vec![
            Value::symbol("llvm.store"),
            value.clone(),
            gep,
        ]);

        Ok(store)
    }

    fn doc(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_field_get() {
        let macro_impl = StructFieldGetMacro::new(
            "Point/x",
            "!llvm.struct<(i64, i64)>",
            0,
            "i64",
        );

        let args = vec![Value::symbol("p")];
        let result = macro_impl.expand(&args).unwrap();

        // Should produce (llvm.load {...} (llvm.getelementptr {...} p))
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "llvm.load");
            } else {
                panic!("Expected llvm.load symbol");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_struct_field_set() {
        let macro_impl = StructFieldSetMacro::new(
            "Point/x!",
            "!llvm.struct<(i64, i64)>",
            0,
        );

        let args = vec![Value::symbol("p"), Value::Number(42.0)];
        let result = macro_impl.expand(&args).unwrap();

        // Should produce (llvm.store 42 (llvm.getelementptr {...} p))
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "llvm.store");
            } else {
                panic!("Expected llvm.store symbol");
            }
        } else {
            panic!("Expected list");
        }
    }
}
