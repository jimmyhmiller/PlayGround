//! Pointer operation macros for high-level syntax
//!
//! null-ptr   - Create null pointer: (null-ptr) -> (llvm.mlir.zero {:result !llvm.ptr})
//! null?      - Check if null: (null? ptr) -> (llvm.icmp {:predicate 0} ptr (llvm.mlir.zero {:result !llvm.ptr}))
//! ptr-load   - Load from pointer: (ptr-load TYPE ptr) -> (llvm.load {:result TYPE} ptr)
//! ptr-store! - Store to pointer: (ptr-store! val ptr) -> (llvm.store val ptr)
//! ptr-offset - Pointer arithmetic with CONSTANT offset: (ptr-offset TYPE ptr offset) -> (llvm.getelementptr {...} ptr)
//! ptr-at     - Pointer arithmetic with DYNAMIC index: (ptr-at ELEM_TYPE ptr idx) -> (llvm.getelementptr {...} ptr idx)

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Null pointer: (null-ptr) -> (llvm.mlir.zero {:result !llvm.ptr})
pub struct NullPtrMacro;

impl Macro for NullPtrMacro {
    fn name(&self) -> &str {
        "null-ptr"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if !args.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "null-ptr".into(),
                expected: "0".into(),
                got: args.len(),
            });
        }

        let mut attrs = HashMap::new();
        attrs.insert("result".to_string(), Value::symbol("!llvm.ptr"));

        Ok(Value::List(vec![
            Value::symbol("llvm.mlir.zero"),
            Value::Map(attrs),
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Create null pointer: (null-ptr) -> (llvm.mlir.zero {:result !llvm.ptr})")
    }
}

/// Null check: (null? ptr) -> (llvm.icmp {:predicate 0} ptr (llvm.mlir.zero {:result !llvm.ptr}))
pub struct NullCheckMacro;

impl Macro for NullCheckMacro {
    fn name(&self) -> &str {
        "null?"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 1 {
            return Err(MacroError::WrongArity {
                macro_name: "null?".into(),
                expected: "1".into(),
                got: args.len(),
            });
        }

        // Build (llvm.mlir.zero {:result !llvm.ptr})
        let mut zero_attrs = HashMap::new();
        zero_attrs.insert("result".to_string(), Value::symbol("!llvm.ptr"));
        let null_ptr = Value::List(vec![
            Value::symbol("llvm.mlir.zero"),
            Value::Map(zero_attrs),
        ]);

        // Build (llvm.icmp {:predicate 0} ptr null)
        // predicate 0 = eq
        let mut cmp_attrs = HashMap::new();
        cmp_attrs.insert("predicate".to_string(), Value::Number(0.0));

        Ok(Value::List(vec![
            Value::symbol("llvm.icmp"),
            Value::Map(cmp_attrs),
            args[0].clone(),
            null_ptr,
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Check if pointer is null: (null? ptr) -> (llvm.icmp {:predicate 0} ptr (null-ptr))")
    }
}

/// Load from pointer: (ptr-load TYPE ptr) -> (llvm.load {:result TYPE} ptr)
pub struct PtrLoadMacro;

impl Macro for PtrLoadMacro {
    fn name(&self) -> &str {
        "ptr-load"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 2 {
            return Err(MacroError::WrongArity {
                macro_name: "ptr-load".into(),
                expected: "2 (TYPE, ptr)".into(),
                got: args.len(),
            });
        }

        let mut attrs = HashMap::new();
        attrs.insert("result".to_string(), args[0].clone());

        Ok(Value::List(vec![
            Value::symbol("llvm.load"),
            Value::Map(attrs),
            args[1].clone(),
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Load from pointer: (ptr-load TYPE ptr) -> (llvm.load {:result TYPE} ptr)")
    }
}

/// Store to pointer: (ptr-store! val ptr) -> (llvm.store val ptr)
pub struct PtrStoreMacro;

impl Macro for PtrStoreMacro {
    fn name(&self) -> &str {
        "ptr-store!"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 2 {
            return Err(MacroError::WrongArity {
                macro_name: "ptr-store!".into(),
                expected: "2 (value, ptr)".into(),
                got: args.len(),
            });
        }

        Ok(Value::List(vec![
            Value::symbol("llvm.store"),
            args[0].clone(),
            args[1].clone(),
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Store to pointer: (ptr-store! val ptr) -> (llvm.store val ptr)")
    }
}

/// Pointer offset: (ptr-offset TYPE ptr offset) -> (llvm.getelementptr {...} ptr)
pub struct PtrOffsetMacro;

impl Macro for PtrOffsetMacro {
    fn name(&self) -> &str {
        "ptr-offset"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 3 {
            return Err(MacroError::WrongArity {
                macro_name: "ptr-offset".into(),
                expected: "3 (TYPE, ptr, offset)".into(),
                got: args.len(),
            });
        }

        let result_type = args[0].clone();
        let ptr = args[1].clone();
        let offset = match &args[2] {
            Value::Number(n) => *n as i32,
            Value::List(items) if items.len() == 3 => {
                // Handle (: N type) form - extract the number
                if let Value::Symbol(s) = &items[0] {
                    if s.name == ":" {
                        if let Value::Number(n) = &items[1] {
                            *n as i32
                        } else {
                            return Err(MacroError::TypeError {
                                macro_name: "ptr-offset".into(),
                                expected: "number for offset",
                                got: format!("{:?}", args[2]),
                            });
                        }
                    } else {
                        return Err(MacroError::TypeError {
                            macro_name: "ptr-offset".into(),
                            expected: "number or (: N type) for offset",
                            got: format!("{:?}", args[2]),
                        });
                    }
                } else {
                    return Err(MacroError::TypeError {
                        macro_name: "ptr-offset".into(),
                        expected: "number or (: N type) for offset",
                        got: format!("{:?}", args[2]),
                    });
                }
            }
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "ptr-offset".into(),
                    expected: "number for offset",
                    got: format!("{:?}", args[2]),
                })
            }
        };

        // Build {:result TYPE :rawConstantIndices array<i32: OFFSET> :elem_type i8}
        let mut attrs = HashMap::new();
        attrs.insert("result".to_string(), result_type);
        attrs.insert(
            "rawConstantIndices".to_string(),
            Value::symbol(&format!("array<i32: {}>", offset)),
        );
        attrs.insert("elem_type".to_string(), Value::symbol("i8"));

        Ok(Value::List(vec![
            Value::symbol("llvm.getelementptr"),
            Value::Map(attrs),
            ptr,
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Pointer offset: (ptr-offset TYPE ptr offset) -> (llvm.getelementptr {...} ptr)")
    }
}

/// Dynamic pointer indexing: (ptr-at ELEM_TYPE ptr idx) -> (llvm.getelementptr {...} ptr idx)
/// Uses dynamic index (passed as operand) rather than constant index.
///
/// Example:
///   (ptr-at f32 data_ptr i) ; where i is a dynamic i64 index
///
/// This is necessary for array indexing where the index is computed at runtime,
/// such as in GPT-2 forward pass where we need: params[token * 768 + c]
pub struct PtrAtMacro;

impl Macro for PtrAtMacro {
    fn name(&self) -> &str {
        "ptr-at"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 3 {
            return Err(MacroError::WrongArity {
                macro_name: "ptr-at".into(),
                expected: "3 (ELEM_TYPE, ptr, idx)".into(),
                got: args.len(),
            });
        }

        let elem_type = args[0].clone();
        let ptr = args[1].clone();
        let idx = args[2].clone();

        // Build {:result !llvm.ptr :rawConstantIndices array<i32: -2147483648> :elem_type ELEM_TYPE}
        // The sentinel value -2147483648 (i32::MIN) tells MLIR to use the operand as dynamic index
        let mut attrs = HashMap::new();
        attrs.insert("result".to_string(), Value::symbol("!llvm.ptr"));
        attrs.insert(
            "rawConstantIndices".to_string(),
            Value::symbol("array<i32: -2147483648>"),
        );
        attrs.insert("elem_type".to_string(), elem_type);

        Ok(Value::List(vec![
            Value::symbol("llvm.getelementptr"),
            Value::Map(attrs),
            ptr,
            idx,
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Dynamic pointer indexing: (ptr-at ELEM_TYPE ptr idx) -> (llvm.getelementptr {...} ptr idx)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_ptr() {
        let macro_impl = NullPtrMacro;
        let result = macro_impl.expand(&[]).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].as_symbol().name, "llvm.mlir.zero");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_null_check() {
        let macro_impl = NullCheckMacro;
        let args = vec![Value::symbol("ptr")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0].as_symbol().name, "llvm.icmp");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_ptr_load() {
        let macro_impl = PtrLoadMacro;
        let args = vec![Value::symbol("!llvm.ptr"), Value::symbol("node")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "llvm.load");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_ptr_store() {
        let macro_impl = PtrStoreMacro;
        let args = vec![Value::symbol("val"), Value::symbol("ptr")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "llvm.store");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_ptr_offset() {
        let macro_impl = PtrOffsetMacro;
        let args = vec![
            Value::symbol("!llvm.ptr"),
            Value::symbol("node"),
            Value::Number(8.0),
        ];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "llvm.getelementptr");
            if let Value::Map(m) = &items[1] {
                if let Some(Value::Symbol(s)) = m.get("rawConstantIndices") {
                    assert_eq!(s.name, "array<i32: 8>");
                } else {
                    panic!("Expected rawConstantIndices");
                }
            } else {
                panic!("Expected map");
            }
        } else {
            panic!("Expected list");
        }
    }
}
