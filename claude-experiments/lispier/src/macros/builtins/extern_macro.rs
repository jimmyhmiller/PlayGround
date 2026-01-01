//! External function declaration macro
//!
//! extern-fn - Declare external function: (extern-fn name (-> [args...] [rets...]))
//!             -> (func.func {:sym_name "name" :function_type (-> ...) :sym_visibility "private"})
//!
//! For variadic functions (with ... in args):
//!   (extern-fn printf (-> [!llvm.ptr ...] [i32]))
//!   -> (llvm.func {:sym_name "printf" :function_type (-> [!llvm.ptr ...] [i32]) :linkage 10})
//!
//! Note: Named extern-fn (not extern) to avoid conflict with the built-in (extern :value-ffi) form.
//!
//! Example:
//!   (extern-fn malloc (-> [i64] [!llvm.ptr]))
//!   -> (func.func {:sym_name "malloc" :function_type (-> [i64] [!llvm.ptr]) :sym_visibility "private"})

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Check if a function type value contains ... (vararg) and extract the args/rets
fn parse_arrow_type(func_type: &Value) -> Option<(Vec<String>, Vec<String>, bool)> {
    if let Value::List(items) = func_type {
        // (-> [args...] [rets...])
        if items.len() >= 3 {
            if let Value::Symbol(sym) = &items[0] {
                if sym.name == "->" {
                    if let Value::Vector(args) = &items[1] {
                        if let Value::Vector(rets) = &items[2] {
                            let mut is_vararg = false;
                            let mut arg_types = Vec::new();
                            for arg in args {
                                if let Value::Symbol(s) = arg {
                                    if s.name == "..." {
                                        is_vararg = true;
                                    } else {
                                        // Convert !llvm.ptr to ptr for LLVM function type syntax
                                        let t = if s.name == "!llvm.ptr" { "ptr".to_string() } else { s.name.clone() };
                                        arg_types.push(t);
                                    }
                                }
                            }
                            let ret_types: Vec<String> = rets.iter().filter_map(|r| {
                                if let Value::Symbol(s) = r {
                                    Some(s.name.clone())
                                } else {
                                    None
                                }
                            }).collect();
                            return Some((arg_types, ret_types, is_vararg));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Check if a function type value contains ... (vararg)
fn is_vararg_type(func_type: &Value) -> bool {
    parse_arrow_type(func_type).map(|(_, _, is_vararg)| is_vararg).unwrap_or(false)
}

/// External function declaration: (extern-fn name fn-type)
pub struct ExternMacro;

impl Macro for ExternMacro {
    fn name(&self) -> &str {
        "extern-fn"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 2 {
            return Err(MacroError::WrongArity {
                macro_name: "extern-fn".into(),
                expected: "2 (name, function-type)".into(),
                got: args.len(),
            });
        }

        let func_name = match &args[0] {
            Value::Symbol(sym) => sym.name.clone(),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "extern-fn".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", args[0]),
                })
            }
        };

        let func_type = args[1].clone();
        let is_vararg = is_vararg_type(&func_type);

        if is_vararg {
            // For vararg functions, use llvm.func with proper LLVM function type syntax
            // LLVM 21+ requires an empty region even for declarations
            let mut attrs = HashMap::new();
            attrs.insert("sym_name".to_string(), Value::String(func_name));

            // Convert (-> [args ...] [ret]) to !llvm.func<ret (args, ...)>
            if let Some((arg_types, ret_types, _)) = parse_arrow_type(&func_type) {
                let args_str = if arg_types.is_empty() {
                    "...".to_string()
                } else {
                    format!("{}, ...", arg_types.join(", "))
                };
                let ret_str = if ret_types.is_empty() {
                    "void".to_string()
                } else {
                    ret_types.join(", ")
                };
                let llvm_func_type = format!("!llvm.func<{} ({})>", ret_str, args_str);
                attrs.insert("function_type".to_string(), Value::symbol(&llvm_func_type));
            } else {
                attrs.insert("function_type".to_string(), func_type);
            }

            // Use proper linkage syntax
            attrs.insert("linkage".to_string(), Value::String("#llvm.linkage<external>".to_string()));

            Ok(Value::List(vec![
                Value::symbol("llvm.func"),
                Value::Map(attrs),
                // Add an empty region (required by LLVM 21+)
                Value::List(vec![
                    Value::symbol("region"),
                ]),
            ]))
        } else {
            // For regular functions, use func.func
            let mut attrs = HashMap::new();
            attrs.insert("sym_name".to_string(), Value::String(func_name));
            attrs.insert("function_type".to_string(), func_type);
            attrs.insert(
                "sym_visibility".to_string(),
                Value::String("private".to_string()),
            );

            Ok(Value::List(vec![
                Value::symbol("func.func"),
                Value::Map(attrs),
            ]))
        }
    }

    fn doc(&self) -> Option<&str> {
        Some("Declare external function: (extern-fn name (-> [args] [rets])) -> (func.func or llvm.func for vararg)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extern_fn() {
        let macro_impl = ExternMacro;
        // (extern-fn malloc (-> [i64] [!llvm.ptr]))
        let fn_type = Value::List(vec![
            Value::symbol("->"),
            Value::Vector(vec![Value::symbol("i64")]),
            Value::Vector(vec![Value::symbol("!llvm.ptr")]),
        ]);
        let args = vec![Value::symbol("malloc"), fn_type];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].as_symbol().name, "func.func");

            if let Value::Map(m) = &items[1] {
                if let Some(Value::String(name)) = m.get("sym_name") {
                    assert_eq!(name, "malloc");
                } else {
                    panic!("Expected sym_name string");
                }
                if let Some(Value::String(vis)) = m.get("sym_visibility") {
                    assert_eq!(vis, "private");
                } else {
                    panic!("Expected sym_visibility string");
                }
                assert!(m.contains_key("function_type"));
            } else {
                panic!("Expected map");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_extern_fn_wrong_arity() {
        let macro_impl = ExternMacro;
        let args = vec![Value::symbol("malloc")];
        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
