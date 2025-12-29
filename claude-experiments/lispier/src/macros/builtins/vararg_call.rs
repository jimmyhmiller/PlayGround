//! Vararg function call macro for calling variadic C functions
//!
//! vararg-call - Call variadic function: (vararg-call TYPE func-name func-type args...)
//!               -> (llvm.call {:callee @func-name :vararg func-type :result TYPE} args...)
//!
//! Example:
//!   (vararg-call i32 printf (-> [!llvm.ptr ...] [i32]) fmt_str arg1 arg2)
//!   -> (llvm.call {:callee @printf :vararg (-> [!llvm.ptr ...] [i32]) :result i32} fmt_str arg1 arg2)

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Vararg function call: (vararg-call TYPE func func-type args...) -> (llvm.call {:callee @func :vararg func-type :result TYPE} args...)
pub struct VarargCallMacro;

impl Macro for VarargCallMacro {
    fn name(&self) -> &str {
        "vararg-call"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() < 3 {
            return Err(MacroError::WrongArity {
                macro_name: "vararg-call".into(),
                expected: "at least 3 (TYPE, func-name, func-type, args...)".into(),
                got: args.len(),
            });
        }

        let result_type = args[0].clone();
        let func_name = match &args[1] {
            Value::Symbol(sym) => format!("@{}", sym.name),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "vararg-call".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", args[1]),
                })
            }
        };
        let func_type = args[2].clone();

        let mut attrs = HashMap::new();
        attrs.insert("callee".to_string(), Value::symbol(&func_name));
        attrs.insert("vararg".to_string(), func_type);
        attrs.insert("result".to_string(), result_type);

        let mut call = vec![Value::symbol("llvm.call"), Value::Map(attrs)];
        call.extend(args[3..].iter().cloned());

        Ok(Value::List(call))
    }

    fn doc(&self) -> Option<&str> {
        Some("Vararg function call: (vararg-call TYPE func func-type args...) -> (llvm.call with vararg attribute)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vararg_call() {
        let macro_impl = VarargCallMacro;
        // (vararg-call i32 printf (-> [!llvm.ptr ...] [i32]) fmt arg)
        let fn_type = Value::List(vec![
            Value::symbol("->"),
            Value::Vector(vec![Value::symbol("!llvm.ptr"), Value::symbol("...")]),
            Value::Vector(vec![Value::symbol("i32")]),
        ]);
        let args = vec![
            Value::symbol("i32"),
            Value::symbol("printf"),
            fn_type,
            Value::symbol("fmt"),
            Value::symbol("arg"),
        ];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 4); // llvm.call, attrs, fmt, arg
            assert_eq!(items[0].as_symbol().name, "llvm.call");

            if let Value::Map(m) = &items[1] {
                if let Some(Value::Symbol(callee)) = m.get("callee") {
                    assert_eq!(callee.name, "@printf");
                } else {
                    panic!("Expected callee symbol");
                }
                assert!(m.contains_key("vararg"));
                assert!(m.contains_key("result"));
            } else {
                panic!("Expected map");
            }

            assert_eq!(items[2].as_symbol().name, "fmt");
            assert_eq!(items[3].as_symbol().name, "arg");
        } else {
            panic!("Expected list");
        }
    }
}
