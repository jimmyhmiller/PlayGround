//! Print macros for string output
//!
//! print    - Printf-style printing: (print "fmt" args...) -> printf call
//! println  - Print with newline: (println "text") -> (print "text\n")
//! print-i64 - Print integer: (print-i64 n) -> (print "%ld" n)
//!
//! These macros expand to `__print_internal__` which is later processed
//! by the StringCollector pass to generate LLVM global string constants
//! and proper printf calls.

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Print macro: (print "fmt" args...) -> (__print_internal__ "fmt" args...)
///
/// The `__print_internal__` form is processed by StringCollector after
/// macro expansion to generate LLVM globals and printf calls.
pub struct PrintMacro;

impl Macro for PrintMacro {
    fn name(&self) -> &str {
        "print"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "print".into(),
                expected: "at least 1 (format-string, args...)".into(),
                got: 0,
            });
        }

        // First argument should be a string literal
        match &args[0] {
            Value::String(_) => {}
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "print".into(),
                    expected: "string literal as first argument",
                    got: format!("{:?}", args[0]),
                });
            }
        }

        // Transform to (__print_internal__ "fmt" args...)
        let mut result = vec![Value::symbol("__print_internal__")];
        result.extend(args.iter().cloned());
        Ok(Value::List(result))
    }

    fn doc(&self) -> Option<&str> {
        Some("Printf-style printing: (print \"fmt\" args...) -> printf(fmt, args...)")
    }
}

/// Println macro: (println "text") -> (print "text\n")
///
/// Convenience macro that appends a newline to the format string.
pub struct PrintlnMacro;

impl Macro for PrintlnMacro {
    fn name(&self) -> &str {
        "println"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.is_empty() {
            // (println) with no args prints just a newline
            return Ok(Value::List(vec![
                Value::symbol("__print_internal__"),
                Value::String("\n".to_string()),
            ]));
        }

        // First argument should be a string literal
        let fmt = match &args[0] {
            Value::String(s) => s.clone(),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "println".into(),
                    expected: "string literal as first argument",
                    got: format!("{:?}", args[0]),
                });
            }
        };

        // Append newline to the format string
        let fmt_with_newline = format!("{}\n", fmt);

        // Transform to (__print_internal__ "fmt\n" args...)
        let mut result = vec![
            Value::symbol("__print_internal__"),
            Value::String(fmt_with_newline),
        ];
        result.extend(args[1..].iter().cloned());
        Ok(Value::List(result))
    }

    fn doc(&self) -> Option<&str> {
        Some("Print with newline: (println \"text\") -> printf(\"text\\n\")")
    }
}

/// Print-i64 macro: (print-i64 n) -> (print "%ld" n)
///
/// Convenience macro for printing a single i64 value.
pub struct PrintI64Macro;

impl Macro for PrintI64Macro {
    fn name(&self) -> &str {
        "print-i64"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 1 {
            return Err(MacroError::WrongArity {
                macro_name: "print-i64".into(),
                expected: "1".into(),
                got: args.len(),
            });
        }

        // Transform to (__print_internal__ "%ld" n)
        Ok(Value::List(vec![
            Value::symbol("__print_internal__"),
            Value::String("%ld".to_string()),
            args[0].clone(),
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Print integer: (print-i64 n) -> printf(\"%ld\", n)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_basic() {
        let macro_impl = PrintMacro;
        let args = vec![Value::String("hello".into())];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].as_symbol().name, "__print_internal__");
            if let Value::String(s) = &items[1] {
                assert_eq!(s, "hello");
            } else {
                panic!("Expected string");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_print_with_args() {
        let macro_impl = PrintMacro;
        let args = vec![
            Value::String("x=%ld".into()),
            Value::symbol("x"),
        ];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "__print_internal__");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_println() {
        let macro_impl = PrintlnMacro;
        let args = vec![Value::String("hello".into())];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
            if let Value::String(s) = &items[1] {
                assert_eq!(s, "hello\n");
            } else {
                panic!("Expected string");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_print_i64() {
        let macro_impl = PrintI64Macro;
        let args = vec![Value::symbol("x")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "__print_internal__");
            if let Value::String(s) = &items[1] {
                assert_eq!(s, "%ld");
            } else {
                panic!("Expected string");
            }
        } else {
            panic!("Expected list");
        }
    }
}
