//! JavaScript to Lisp transpiler
//!
//! Converts JavaScript to the Lisp AST used by the partial evaluator.

use swc_common::sync::Lrc;
use swc_common::{FileName, SourceMap};
use swc_ecma_ast::*;
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

use crate::ast::{BinOp, Expr};

/// Parse JavaScript source code and convert it to our Lisp AST
pub fn js_to_lisp(source: &str) -> Result<Expr, String> {
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(FileName::Custom("input.js".into()).into(), source.into());

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        Default::default(),
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    let script = parser
        .parse_script()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    convert_statements(&script.body)
}

/// Convert a list of statements to a single expression
fn convert_statements(stmts: &[Stmt]) -> Result<Expr, String> {
    if stmts.is_empty() {
        return Ok(Expr::Undefined);
    }

    if stmts.len() == 1 {
        return convert_stmt(&stmts[0]);
    }

    // Multiple statements - handle declarations that need to scope over rest
    let mut exprs = Vec::new();
    let mut rest_stmts = stmts.to_vec();

    while !rest_stmts.is_empty() {
        let stmt = &rest_stmts[0];

        if let Stmt::Decl(Decl::Var(var_decl)) = stmt {
            let remaining = rest_stmts[1..].to_vec();
            let body = if remaining.is_empty() {
                Expr::Undefined
            } else {
                convert_statements(&remaining)?
            };

            let let_expr = convert_var_decl(var_decl, body)?;

            // If we have collected statements before this declaration,
            // wrap them in a Begin with the Let
            if exprs.is_empty() {
                return Ok(let_expr);
            } else {
                exprs.push(let_expr);
                return Ok(Expr::Begin(exprs));
            }
        }

        if let Stmt::Decl(Decl::Fn(fn_decl)) = stmt {
            let remaining = rest_stmts[1..].to_vec();
            let body = if remaining.is_empty() {
                Expr::Var(fn_decl.ident.sym.to_string())
            } else {
                convert_statements(&remaining)?
            };

            let name = fn_decl.ident.sym.to_string();
            let func = convert_function(&fn_decl.function)?;
            let let_expr = Expr::Let(name, Box::new(func), Box::new(body));

            // If we have collected statements before this declaration,
            // wrap them in a Begin with the Let
            if exprs.is_empty() {
                return Ok(let_expr);
            } else {
                exprs.push(let_expr);
                return Ok(Expr::Begin(exprs));
            }
        }

        // Not a var decl, convert normally
        exprs.push(convert_stmt(stmt)?);
        rest_stmts = rest_stmts[1..].to_vec();
    }

    if exprs.len() == 1 {
        Ok(exprs.pop().unwrap())
    } else {
        Ok(Expr::Begin(exprs))
    }
}

/// Convert a variable declaration to nested lets
fn convert_var_decl(var_decl: &VarDecl, body: Expr) -> Result<Expr, String> {
    let mut result = body;

    // Process declarations in reverse order to nest them correctly
    for decl in var_decl.decls.iter().rev() {
        let name = match &decl.name {
            Pat::Ident(ident) => ident.id.sym.to_string(),
            _ => return Err("Only simple variable names are supported".to_string()),
        };

        let init = match &decl.init {
            Some(init) => convert_expr(init)?,
            None => Expr::Undefined,
        };

        result = Expr::Let(name, Box::new(init), Box::new(result));
    }

    Ok(result)
}

/// Convert a statement to an expression
fn convert_stmt(stmt: &Stmt) -> Result<Expr, String> {
    match stmt {
        Stmt::Expr(expr_stmt) => convert_expr(&expr_stmt.expr),

        Stmt::Decl(decl) => match decl {
            Decl::Var(var_decl) => {
                if var_decl.decls.is_empty() {
                    return Ok(Expr::Undefined);
                }
                let last_decl = var_decl.decls.last().unwrap();
                match &last_decl.init {
                    Some(init) => convert_expr(init),
                    None => Ok(Expr::Undefined),
                }
            }
            Decl::Fn(fn_decl) => {
                let name = fn_decl.ident.sym.to_string();
                let func = convert_function(&fn_decl.function)?;
                Ok(Expr::Let(
                    name.clone(),
                    Box::new(func),
                    Box::new(Expr::Var(name)),
                ))
            }
            _ => Err(format!("Unsupported declaration: {:?}", decl)),
        },

        Stmt::Block(block) => convert_statements(&block.stmts),

        Stmt::Return(ret) => match &ret.arg {
            Some(arg) => Ok(Expr::Return(Box::new(convert_expr(arg)?))),
            None => Ok(Expr::Return(Box::new(Expr::Undefined))),
        },

        Stmt::If(if_stmt) => {
            let cond = convert_expr(&if_stmt.test)?;
            let then_branch = convert_stmt(&if_stmt.cons)?;
            let else_branch = match &if_stmt.alt {
                Some(alt) => convert_stmt(alt)?,
                None => Expr::Undefined,
            };
            Ok(Expr::If(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
            ))
        }

        Stmt::While(while_stmt) => {
            let cond = convert_expr(&while_stmt.test)?;
            let body = convert_stmt(&while_stmt.body)?;
            Ok(Expr::While(Box::new(cond), Box::new(body)))
        }

        Stmt::For(for_stmt) => {
            let init = match &for_stmt.init {
                Some(VarDeclOrExpr::VarDecl(var_decl)) => {
                    // Convert var decl to a let-binding series
                    Some(Box::new(convert_var_decl(var_decl, Expr::Undefined)?))
                }
                Some(VarDeclOrExpr::Expr(expr)) => Some(Box::new(convert_expr(expr)?)),
                None => None,
            };

            let cond = match &for_stmt.test {
                Some(test) => Some(Box::new(convert_expr(test)?)),
                None => None,
            };

            let update = match &for_stmt.update {
                Some(update) => Some(Box::new(convert_expr(update)?)),
                None => None,
            };

            let body = convert_stmt(&for_stmt.body)?;

            Ok(Expr::For {
                init,
                cond,
                update,
                body: Box::new(body),
            })
        }

        Stmt::Switch(switch_stmt) => {
            let discriminant = convert_expr(&switch_stmt.discriminant)?;
            let mut cases = Vec::new();
            let mut default = None;

            for case in &switch_stmt.cases {
                // Use convert_statements to handle var declarations in case bodies
                let body = if case.cons.is_empty() {
                    vec![]
                } else {
                    vec![convert_statements(&case.cons)?]
                };

                if let Some(test) = &case.test {
                    let test_expr = convert_expr(test)?;
                    cases.push((test_expr, body));
                } else {
                    default = Some(body);
                }
            }

            Ok(Expr::Switch {
                discriminant: Box::new(discriminant),
                cases,
                default,
            })
        }

        Stmt::Break(_) => Ok(Expr::Break),
        Stmt::Continue(_) => Ok(Expr::Continue),

        Stmt::Throw(throw_stmt) => {
            let arg = convert_expr(&throw_stmt.arg)?;
            Ok(Expr::Throw(Box::new(arg)))
        }

        Stmt::Empty(_) => Ok(Expr::Undefined),

        Stmt::Try(try_stmt) => {
            // Convert try block - use convert_statements to handle var declarations
            let try_block = convert_statements(&try_stmt.block.stmts)?;

            // Convert catch handler
            let (catch_param, catch_block) = if let Some(handler) = &try_stmt.handler {
                let param = handler.param.as_ref().map(|p| {
                    if let swc_ecma_ast::Pat::Ident(ident) = p {
                        ident.id.sym.to_string()
                    } else {
                        "_error".to_string()
                    }
                });
                let catch_body = convert_statements(&handler.body.stmts)?;
                (param, catch_body)
            } else {
                (None, Expr::Undefined)
            };

            // Convert finally block
            let finally_block = if let Some(finalizer) = &try_stmt.finalizer {
                Some(Box::new(convert_statements(&finalizer.stmts)?))
            } else {
                None
            };

            Ok(Expr::TryCatch {
                try_block: Box::new(try_block),
                catch_param,
                catch_block: Box::new(catch_block),
                finally_block,
            })
        }

        _ => Err(format!("Unsupported statement: {:?}", stmt)),
    }
}

/// Convert a JavaScript expression to our Lisp AST
fn convert_expr(expr: &swc_ecma_ast::Expr) -> Result<Expr, String> {
    match expr {
        // Literals
        swc_ecma_ast::Expr::Lit(lit) => convert_lit(lit),

        // Variables
        swc_ecma_ast::Expr::Ident(ident) => {
            let name = ident.sym.to_string();
            match name.as_str() {
                "undefined" => Ok(Expr::Undefined),
                "null" => Ok(Expr::Null),
                _ => Ok(Expr::Var(name)),
            }
        }

        // Binary operations
        swc_ecma_ast::Expr::Bin(bin) => {
            let left = convert_expr(&bin.left)?;
            let right = convert_expr(&bin.right)?;
            let op = convert_binop(bin.op)?;
            Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
        }

        // Unary operations
        swc_ecma_ast::Expr::Unary(unary) => {
            let arg = convert_expr(&unary.arg)?;
            match unary.op {
                UnaryOp::Minus => {
                    // -n for numeric literal becomes just the negative number
                    if let Expr::Int(n) = arg {
                        Ok(Expr::Int(-n))
                    } else {
                        // -x becomes (- 0 x)
                        Ok(Expr::BinOp(
                            BinOp::Sub,
                            Box::new(Expr::Int(0)),
                            Box::new(arg),
                        ))
                    }
                }
                UnaryOp::Plus => {
                    // +x is just x (type coercion, ignore for now)
                    Ok(arg)
                }
                UnaryOp::Bang => Ok(Expr::LogNot(Box::new(arg))),
                UnaryOp::Tilde => Ok(Expr::BitNot(Box::new(arg))),
                UnaryOp::TypeOf => {
                    // typeof x - treat as opaque
                    Ok(Expr::Opaque(format!("typeof {}", arg)))
                }
                UnaryOp::Void => {
                    // void x returns undefined
                    Ok(Expr::Begin(vec![arg, Expr::Undefined]))
                }
                _ => Err(format!("Unsupported unary operator: {:?}", unary.op)),
            }
        }

        // Ternary conditional: cond ? then : else
        swc_ecma_ast::Expr::Cond(cond) => {
            let test = convert_expr(&cond.test)?;
            let cons = convert_expr(&cond.cons)?;
            let alt = convert_expr(&cond.alt)?;
            Ok(Expr::If(Box::new(test), Box::new(cons), Box::new(alt)))
        }

        // Parenthesized expression
        swc_ecma_ast::Expr::Paren(paren) => convert_expr(&paren.expr),

        // Sequence expression: (a, b, c) -> (begin a b c)
        swc_ecma_ast::Expr::Seq(seq) => {
            let exprs: Result<Vec<Expr>, String> =
                seq.exprs.iter().map(|e| convert_expr(e)).collect();
            Ok(Expr::Begin(exprs?))
        }

        // Arrow function: (x, y) => body
        swc_ecma_ast::Expr::Arrow(arrow) => {
            let params: Result<Vec<String>, String> = arrow
                .params
                .iter()
                .map(|p| match p {
                    Pat::Ident(ident) => Ok(ident.id.sym.to_string()),
                    _ => Err("Only simple parameter names are supported".to_string()),
                })
                .collect();

            let body = match &*arrow.body {
                BlockStmtOrExpr::BlockStmt(block) => convert_statements(&block.stmts)?,
                BlockStmtOrExpr::Expr(expr) => convert_expr(expr)?,
            };

            Ok(Expr::Fn(params?, Box::new(body)))
        }

        // Function expression: function(x, y) { body }
        swc_ecma_ast::Expr::Fn(fn_expr) => convert_function(&fn_expr.function),

        // Function call: f(a, b, c)
        swc_ecma_ast::Expr::Call(call) => {
            let callee = match &call.callee {
                Callee::Expr(expr) => convert_expr(expr)?,
                _ => return Err("Only expression callees are supported".to_string()),
            };

            let args: Result<Vec<Expr>, String> = call
                .args
                .iter()
                .map(|arg| {
                    if arg.spread.is_some() {
                        Err("Spread arguments are not supported".to_string())
                    } else {
                        convert_expr(&arg.expr)
                    }
                })
                .collect();

            Ok(Expr::Call(Box::new(callee), args?))
        }

        // New expression: new Constructor(args)
        swc_ecma_ast::Expr::New(new_expr) => {
            let callee = convert_expr(&new_expr.callee)?;
            let args: Result<Vec<Expr>, String> = new_expr
                .args
                .as_ref()
                .map(|args| {
                    args.iter()
                        .map(|arg| {
                            if arg.spread.is_some() {
                                Err("Spread arguments are not supported".to_string())
                            } else {
                                convert_expr(&arg.expr)
                            }
                        })
                        .collect()
                })
                .unwrap_or_else(|| Ok(Vec::new()));

            Ok(Expr::New(Box::new(callee), args?))
        }

        // Array literal: [a, b, c]
        swc_ecma_ast::Expr::Array(arr) => {
            let elems: Result<Vec<Expr>, String> = arr
                .elems
                .iter()
                .map(|elem| match elem {
                    Some(ExprOrSpread { spread: None, expr }) => convert_expr(expr),
                    Some(ExprOrSpread {
                        spread: Some(_), ..
                    }) => Err("Spread elements are not supported".to_string()),
                    None => Ok(Expr::Undefined), // Hole in array
                })
                .collect();
            Ok(Expr::Array(elems?))
        }

        // Object literal: {a: 1, b: 2}
        swc_ecma_ast::Expr::Object(obj) => {
            let mut props = Vec::new();

            for prop in &obj.props {
                match prop {
                    PropOrSpread::Prop(prop) => {
                        match &**prop {
                            Prop::KeyValue(kv) => {
                                let key = match &kv.key {
                                    PropName::Ident(ident) => ident.sym.to_string(),
                                    PropName::Str(s) => s.value.to_string(),
                                    PropName::Num(n) => n.value.to_string(),
                                    _ => {
                                        return Err(
                                            "Only identifier/string/number keys are supported"
                                                .to_string(),
                                        )
                                    }
                                };
                                let value = convert_expr(&kv.value)?;
                                props.push((key, value));
                            }
                            Prop::Shorthand(ident) => {
                                // {x} is shorthand for {x: x}
                                let name = ident.sym.to_string();
                                props.push((name.clone(), Expr::Var(name)));
                            }
                            Prop::Method(method) => {
                                let key = match &method.key {
                                    PropName::Ident(ident) => ident.sym.to_string(),
                                    PropName::Str(s) => s.value.to_string(),
                                    _ => return Err("Only identifier/string method names are supported".to_string()),
                                };
                                let func = convert_function(&method.function)?;
                                props.push((key, func));
                            }
                            _ => return Err(format!("Unsupported object property type: {:?}", prop)),
                        }
                    }
                    PropOrSpread::Spread(_) => {
                        return Err("Spread in object literals is not supported".to_string())
                    }
                }
            }

            Ok(Expr::Object(props))
        }

        // Member access: obj.prop or obj[key]
        swc_ecma_ast::Expr::Member(member) => {
            let obj = convert_expr(&member.obj)?;

            match &member.prop {
                // Computed property: obj[expr]
                MemberProp::Computed(computed) => {
                    let prop = convert_expr(&computed.expr)?;
                    // Check if it's an array with numeric index vs object with key
                    Ok(Expr::Index(Box::new(obj), Box::new(prop)))
                }
                // Identifier property: obj.prop
                MemberProp::Ident(ident) => {
                    let prop_name = ident.sym.to_string();
                    if prop_name == "length" {
                        // Special case for .length
                        Ok(Expr::Len(Box::new(obj)))
                    } else {
                        Ok(Expr::PropAccess(Box::new(obj), prop_name))
                    }
                }
                _ => Err("Unsupported member access".to_string()),
            }
        }

        // Assignment: x = value or obj.prop = value
        swc_ecma_ast::Expr::Assign(assign) => {
            let value = convert_expr(&assign.right)?;

            // Handle compound assignment operators
            let value = match assign.op {
                AssignOp::Assign => value,
                AssignOp::AddAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Add, Box::new(left), Box::new(value))
                }
                AssignOp::SubAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Sub, Box::new(left), Box::new(value))
                }
                AssignOp::MulAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Mul, Box::new(left), Box::new(value))
                }
                AssignOp::DivAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Div, Box::new(left), Box::new(value))
                }
                AssignOp::ModAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Mod, Box::new(left), Box::new(value))
                }
                AssignOp::BitAndAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::BitAnd, Box::new(left), Box::new(value))
                }
                AssignOp::BitOrAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::BitOr, Box::new(left), Box::new(value))
                }
                AssignOp::BitXorAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::BitXor, Box::new(left), Box::new(value))
                }
                AssignOp::LShiftAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Shl, Box::new(left), Box::new(value))
                }
                AssignOp::RShiftAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::Shr, Box::new(left), Box::new(value))
                }
                AssignOp::ZeroFillRShiftAssign => {
                    let left = convert_assign_target_to_expr(&assign.left)?;
                    Expr::BinOp(BinOp::UShr, Box::new(left), Box::new(value))
                }
                _ => return Err(format!("Unsupported assignment operator: {:?}", assign.op)),
            };

            match &assign.left {
                AssignTarget::Simple(SimpleAssignTarget::Ident(ident)) => {
                    Ok(Expr::Set(ident.id.sym.to_string(), Box::new(value)))
                }
                AssignTarget::Simple(SimpleAssignTarget::Member(member)) => {
                    let obj = convert_expr(&member.obj)?;
                    match &member.prop {
                        MemberProp::Ident(ident) => {
                            let prop_name = ident.sym.to_string();
                            Ok(Expr::PropSet(Box::new(obj), prop_name, Box::new(value)))
                        }
                        MemberProp::Computed(computed) => {
                            let key = convert_expr(&computed.expr)?;
                            Ok(Expr::ComputedSet(Box::new(obj), Box::new(key), Box::new(value)))
                        }
                        _ => Err("Unsupported member assignment".to_string()),
                    }
                }
                _ => Err("Unsupported assignment target".to_string()),
            }
        }

        // Update expressions: x++, x--, ++x, --x
        swc_ecma_ast::Expr::Update(update) => {
            let name = match &*update.arg {
                swc_ecma_ast::Expr::Ident(ident) => ident.sym.to_string(),
                _ => return Err("Update expressions only support simple variables".to_string()),
            };

            let op = match update.op {
                UpdateOp::PlusPlus => BinOp::Add,
                UpdateOp::MinusMinus => BinOp::Sub,
            };

            let new_value = Expr::BinOp(
                op,
                Box::new(Expr::Var(name.clone())),
                Box::new(Expr::Int(1)),
            );

            if update.prefix {
                // ++x: increment, then return NEW value
                // (begin (set! x (+ x 1)) x)
                Ok(Expr::Begin(vec![
                    Expr::Set(name.clone(), Box::new(new_value)),
                    Expr::Var(name),
                ]))
            } else {
                // x++: save OLD value, increment, return OLD value
                // (let _old_x x (begin (set! x (+ x 1)) _old_x))
                let old_var = format!("_old_{}", name);
                Ok(Expr::Let(
                    old_var.clone(),
                    Box::new(Expr::Var(name.clone())),
                    Box::new(Expr::Begin(vec![
                        Expr::Set(name, Box::new(new_value)),
                        Expr::Var(old_var),
                    ])),
                ))
            }
        }

        // This expression
        swc_ecma_ast::Expr::This(_) => Ok(Expr::Var("this".to_string())),

        _ => Err(format!("Unsupported expression: {:?}", expr)),
    }
}

/// Helper to convert assignment target to expression (for compound assignments)
fn convert_assign_target_to_expr(target: &AssignTarget) -> Result<Expr, String> {
    match target {
        AssignTarget::Simple(SimpleAssignTarget::Ident(ident)) => {
            Ok(Expr::Var(ident.id.sym.to_string()))
        }
        AssignTarget::Simple(SimpleAssignTarget::Member(member)) => {
            let obj = convert_expr(&member.obj)?;
            match &member.prop {
                MemberProp::Ident(ident) => {
                    let prop_name = ident.sym.to_string();
                    if prop_name == "length" {
                        Ok(Expr::Len(Box::new(obj)))
                    } else {
                        Ok(Expr::PropAccess(Box::new(obj), prop_name))
                    }
                }
                MemberProp::Computed(computed) => {
                    let key = convert_expr(&computed.expr)?;
                    Ok(Expr::Index(Box::new(obj), Box::new(key)))
                }
                _ => Err("Unsupported member access in assignment".to_string()),
            }
        }
        _ => Err("Unsupported assignment target".to_string()),
    }
}

/// Convert a function to our Lisp AST
fn convert_function(func: &Function) -> Result<Expr, String> {
    let params: Result<Vec<String>, String> = func
        .params
        .iter()
        .map(|p| match &p.pat {
            Pat::Ident(ident) => Ok(ident.id.sym.to_string()),
            _ => Err("Only simple parameter names are supported".to_string()),
        })
        .collect();

    let body = match &func.body {
        Some(block) => convert_statements(&block.stmts)?,
        None => Expr::Undefined,
    };

    Ok(Expr::Fn(params?, Box::new(body)))
}

/// Convert a JavaScript literal to our Lisp AST
fn convert_lit(lit: &Lit) -> Result<Expr, String> {
    match lit {
        Lit::Num(num) => {
            // Convert to integer (truncate decimals)
            Ok(Expr::Int(num.value as i64))
        }
        Lit::Bool(b) => Ok(Expr::Bool(b.value)),
        Lit::Str(s) => Ok(Expr::String(s.value.to_string())),
        Lit::Null(_) => Ok(Expr::Null),
        _ => Err(format!("Unsupported literal: {:?}", lit)),
    }
}

/// Convert a JavaScript binary operator to our BinOp
fn convert_binop(op: BinaryOp) -> Result<BinOp, String> {
    match op {
        BinaryOp::Add => Ok(BinOp::Add),
        BinaryOp::Sub => Ok(BinOp::Sub),
        BinaryOp::Mul => Ok(BinOp::Mul),
        BinaryOp::Div => Ok(BinOp::Div),
        BinaryOp::Mod => Ok(BinOp::Mod),
        BinaryOp::Lt => Ok(BinOp::Lt),
        BinaryOp::Gt => Ok(BinOp::Gt),
        BinaryOp::LtEq => Ok(BinOp::Lte),
        BinaryOp::GtEq => Ok(BinOp::Gte),
        BinaryOp::EqEq | BinaryOp::EqEqEq => Ok(BinOp::Eq),
        BinaryOp::NotEq | BinaryOp::NotEqEq => Ok(BinOp::NotEq),
        BinaryOp::LogicalAnd => Ok(BinOp::And),
        BinaryOp::LogicalOr => Ok(BinOp::Or),
        BinaryOp::BitAnd => Ok(BinOp::BitAnd),
        BinaryOp::BitOr => Ok(BinOp::BitOr),
        BinaryOp::BitXor => Ok(BinOp::BitXor),
        BinaryOp::LShift => Ok(BinOp::Shl),
        BinaryOp::RShift => Ok(BinOp::Shr),
        BinaryOp::ZeroFillRShift => Ok(BinOp::UShr),
        _ => Err(format!("Unsupported binary operator: {:?}", op)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_number() {
        let result = js_to_lisp("42").unwrap();
        assert_eq!(result, Expr::Int(42));
    }

    #[test]
    fn test_simple_bool() {
        assert_eq!(js_to_lisp("true").unwrap(), Expr::Bool(true));
        assert_eq!(js_to_lisp("false").unwrap(), Expr::Bool(false));
    }

    #[test]
    fn test_variable() {
        assert_eq!(js_to_lisp("x").unwrap(), Expr::Var("x".to_string()));
    }

    #[test]
    fn test_binary_op() {
        let result = js_to_lisp("1 + 2").unwrap();
        assert_eq!(
            result,
            Expr::BinOp(BinOp::Add, Box::new(Expr::Int(1)), Box::new(Expr::Int(2)))
        );
    }

    #[test]
    fn test_bitwise_ops() {
        let result = js_to_lisp("x & 1").unwrap();
        assert!(matches!(result, Expr::BinOp(BinOp::BitAnd, _, _)));

        let result = js_to_lisp("x | 2").unwrap();
        assert!(matches!(result, Expr::BinOp(BinOp::BitOr, _, _)));

        let result = js_to_lisp("x ^ 3").unwrap();
        assert!(matches!(result, Expr::BinOp(BinOp::BitXor, _, _)));

        let result = js_to_lisp("x << 4").unwrap();
        assert!(matches!(result, Expr::BinOp(BinOp::Shl, _, _)));

        let result = js_to_lisp("x >> 5").unwrap();
        assert!(matches!(result, Expr::BinOp(BinOp::Shr, _, _)));
    }

    #[test]
    fn test_object_literal() {
        // Need parentheses to disambiguate from block statement
        let result = js_to_lisp("({a: 1, b: 2})").unwrap();
        assert!(matches!(result, Expr::Object(_)));
    }

    #[test]
    fn test_object_shorthand() {
        let result = js_to_lisp("let x = 1; ({x})").unwrap();
        // Should create an object with x: x
        assert!(matches!(result, Expr::Let(..)));
    }

    #[test]
    fn test_prop_access() {
        let result = js_to_lisp("obj.foo").unwrap();
        assert!(matches!(result, Expr::PropAccess(_, _)));
    }

    #[test]
    fn test_switch() {
        let result = js_to_lisp("switch(x) { case 0: 1; break; case 1: 2; break; }").unwrap();
        assert!(matches!(result, Expr::Switch { .. }));
    }

    #[test]
    fn test_for_loop() {
        let result = js_to_lisp("for (var i = 0; i < 10; i++) { x }").unwrap();
        assert!(matches!(result, Expr::For { .. }));
    }

    #[test]
    fn test_new_expression() {
        let result = js_to_lisp("new Uint8Array(10)").unwrap();
        assert!(matches!(result, Expr::New(_, _)));
    }

    #[test]
    fn test_throw() {
        let result = js_to_lisp("throw x").unwrap();
        assert!(matches!(result, Expr::Throw(_)));
    }
}
