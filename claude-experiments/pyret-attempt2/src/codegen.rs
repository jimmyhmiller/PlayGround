//! Simple Pyret to R4RS Scheme compiler
//!
//! This is a minimal compiler that handles:
//! - Numbers and identifiers
//! - Binary operators (+, -, *, /, <=, >=, <, >, ==)
//! - Function definitions (fun name(args): body end)
//! - Function calls (f(x, y))
//! - If-else expressions
//!
//! Intentionally simple - no objects, data types, pattern matching, etc.

use crate::ast::{BinOp, Expr, Name, Program};

pub struct SchemeCompiler {
    indent_level: usize,
}

impl SchemeCompiler {
    pub fn new() -> Self {
        SchemeCompiler { indent_level: 0 }
    }

    fn indent(&self) -> String {
        "  ".repeat(self.indent_level)
    }

    pub fn compile_program(&mut self, program: &Program) -> Result<String, String> {
        // In Pyret, the program body is a single expression
        // We'll compile it directly
        self.compile_expr(&program.body)
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<String, String> {
        match expr {
            Expr::SNum { value, .. } => Ok(value.clone()),

            Expr::SId { id, .. } => Ok(self.compile_name(id)),

            Expr::SOp { op, left, right, .. } => {
                // Convert Pyret BinOp to Scheme operator
                let op_str = match op {
                    BinOp::Plus => "+",
                    BinOp::Minus => "-",
                    BinOp::Times => "*",
                    BinOp::Divide => "/",
                    BinOp::Leq => "<=",
                    BinOp::Geq => ">=",
                    BinOp::Lt => "<",
                    BinOp::Gt => ">",
                    BinOp::Equal => "=",
                    _ => return Err(format!("Unsupported operator: {:?}", op)),
                };

                let left_str = self.compile_expr(left)?;
                let right_str = self.compile_expr(right)?;

                Ok(format!("({} {} {})", op_str, left_str, right_str))
            }

            Expr::SApp { _fun, args, .. } => {
                let func_str = self.compile_expr(_fun)?;
                let args_str = args
                    .iter()
                    .map(|arg| self.compile_expr(arg))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");

                if args.is_empty() {
                    Ok(format!("({})", func_str))
                } else {
                    Ok(format!("({} {})", func_str, args_str))
                }
            }

            Expr::SIfElse {
                branches,
                _else,
                ..
            } => {
                // Handle simple if-else (Pyret's if has multiple branches)
                // For simplicity, we'll handle the first branch only
                if branches.is_empty() {
                    return Err("If expression has no branches".to_string());
                }

                let first_branch = &branches[0];
                let cond_str = self.compile_expr(&first_branch.test)?;

                self.indent_level += 1;
                let then_str = self.compile_expr(&first_branch.body)?;
                let else_str = self.compile_expr(_else)?;
                self.indent_level -= 1;

                Ok(format!("(if {}\n{}  {}\n{}  {})",
                    cond_str,
                    self.indent(), then_str,
                    self.indent(), else_str))
            }

            Expr::SIf {
                branches,
                ..
            } => {
                // If without else - R4RS requires an else, so use #f
                if branches.is_empty() {
                    return Err("If expression has no branches".to_string());
                }

                let first_branch = &branches[0];
                let cond_str = self.compile_expr(&first_branch.test)?;

                self.indent_level += 1;
                let then_str = self.compile_expr(&first_branch.body)?;
                self.indent_level -= 1;

                Ok(format!("(if {}\n{}  {}\n{}  #f)",
                    cond_str,
                    self.indent(), then_str,
                    self.indent()))
            }

            Expr::SFun {
                name,
                args,
                body,
                ..
            } => {
                let params_str = args
                    .iter()
                    .map(|b| self.compile_bind_name(b))
                    .collect::<Vec<_>>()
                    .join(" ");

                self.indent_level += 1;
                let body_str = self.compile_expr(body)?;
                self.indent_level -= 1;

                Ok(format!("(define ({} {})\n{}  {})",
                    name, params_str,
                    self.indent(), body_str))
            }

            Expr::SBlock { stmts, .. } => {
                // Compile block as a sequence of statements
                if stmts.is_empty() {
                    return Ok("#<void>".to_string());
                }

                if stmts.len() == 1 {
                    return self.compile_expr(&stmts[0]);
                }

                let mut output = String::from("(begin\n");
                self.indent_level += 1;

                for (i, stmt) in stmts.iter().enumerate() {
                    output.push_str(&format!("{}{}", self.indent(), self.compile_expr(stmt)?));
                    if i < stmts.len() - 1 {
                        output.push('\n');
                    }
                }

                self.indent_level -= 1;
                output.push(')');
                Ok(output)
            }

            Expr::SParen { expr, .. } => {
                // Parentheses are just for grouping in Pyret, compile the inner expression
                self.compile_expr(expr)
            }

            _ => Err(format!("Unsupported expression type: {:?}", expr)),
        }
    }

    fn compile_name(&self, name: &Name) -> String {
        match name {
            Name::SName { s, .. } => s.clone(),
            Name::SGlobal { s, .. } => s.clone(),
            Name::SUnderscore { .. } => "_".to_string(),
            Name::SModuleGlobal { s, .. } => s.clone(),
            Name::STypeGlobal { s, .. } => s.clone(),
            Name::SAtom { base, serial, .. } => format!("{}_{}", base, serial),
        }
    }

    fn compile_bind_name(&self, bind: &crate::ast::Bind) -> String {
        match bind {
            crate::ast::Bind::SBind { id, .. } => self.compile_name(id),
            crate::ast::Bind::STupleBind { fields, .. } => {
                // For tuple binds, just use first field for simplicity
                // In a real compiler, you'd destructure
                if let Some(first) = fields.first() {
                    self.compile_bind_name(first)
                } else {
                    "_".to_string()
                }
            }
        }
    }
}

impl Default for SchemeCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn make_loc() -> Loc {
        Loc::new(FileId(0), 0, 0, 0, 0, 0, 0)
    }

    #[test]
    fn test_compile_number() {
        let mut compiler = SchemeCompiler::new();
        let expr = Expr::SNum {
            l: make_loc(),
            value: "42".to_string(),
        };
        assert_eq!(compiler.compile_expr(&expr).unwrap(), "42");
    }

    #[test]
    fn test_compile_id() {
        let mut compiler = SchemeCompiler::new();
        let expr = Expr::SId {
            l: make_loc(),
            id: Name::SName {
                l: make_loc(),
                s: "x".to_string(),
            },
        };
        assert_eq!(compiler.compile_expr(&expr).unwrap(), "x");
    }

    #[test]
    fn test_compile_addition() {
        let mut compiler = SchemeCompiler::new();
        let expr = Expr::SOp {
            l: make_loc(),
            op_l: make_loc(),
            op: BinOp::Plus,
            left: Box::new(Expr::SNum {
                l: make_loc(),
                value: "1".to_string(),
            }),
            right: Box::new(Expr::SNum {
                l: make_loc(),
                value: "2".to_string(),
            }),
        };
        assert_eq!(compiler.compile_expr(&expr).unwrap(), "(+ 1 2)");
    }

    #[test]
    fn test_compile_function_call() {
        let mut compiler = SchemeCompiler::new();
        let expr = Expr::SApp {
            l: make_loc(),
            _fun: Box::new(Expr::SId {
                l: make_loc(),
                id: Name::SName {
                    l: make_loc(),
                    s: "f".to_string(),
                },
            }),
            args: vec![Box::new(Expr::SNum {
                l: make_loc(),
                value: "10".to_string(),
            })],
        };
        assert_eq!(compiler.compile_expr(&expr).unwrap(), "(f 10)");
    }
}
