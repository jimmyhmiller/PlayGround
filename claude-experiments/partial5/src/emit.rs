//! Code generation / emission
//!
//! Converts abstract values and traces back to JavaScript source code.

use swc_ecma_ast::*;

use crate::abstract_value::{AbstractValue, JsValue, FunctionValue};

/// Convert an abstract value to a JavaScript expression
pub fn value_to_expr(value: &AbstractValue) -> Expr {
    match value {
        AbstractValue::Known(js_val) => js_value_to_expr(js_val),
        AbstractValue::Dynamic(name, expr) => {
            if let Some(e) = expr {
                *e.clone()
            } else {
                // Create an identifier reference
                Expr::Ident(Ident {
                    span: Default::default(),
                    ctxt: Default::default(),
                    sym: name.clone().into(),
                    optional: false,
                })
            }
        }
        AbstractValue::Top => {
            // Top means we don't know - emit a placeholder
            Expr::Ident(Ident {
                span: Default::default(),
                ctxt: Default::default(),
                sym: "__unknown__".into(),
                optional: false,
            })
        }
    }
}

/// Convert a known JavaScript value to an expression
fn js_value_to_expr(value: &JsValue) -> Expr {
    match value {
        JsValue::Undefined => Expr::Ident(Ident {
            span: Default::default(),
            ctxt: Default::default(),
            sym: "undefined".into(),
            optional: false,
        }),

        JsValue::Null => Expr::Lit(Lit::Null(Null {
            span: Default::default(),
        })),

        JsValue::Bool(b) => Expr::Lit(Lit::Bool(Bool {
            span: Default::default(),
            value: *b,
        })),

        JsValue::Number(n) => {
            if *n < 0.0 {
                // Negative numbers need to be wrapped
                Expr::Unary(UnaryExpr {
                    span: Default::default(),
                    op: UnaryOp::Minus,
                    arg: Box::new(Expr::Lit(Lit::Num(Number {
                        span: Default::default(),
                        value: -*n,
                        raw: None,
                    }))),
                })
            } else {
                Expr::Lit(Lit::Num(Number {
                    span: Default::default(),
                    value: *n,
                    raw: None,
                }))
            }
        }

        JsValue::String(s) => Expr::Lit(Lit::Str(Str {
            span: Default::default(),
            value: s.clone().into(),
            raw: None,
        })),

        JsValue::Array(arr) => {
            let elems: Vec<Option<ExprOrSpread>> = arr
                .borrow()
                .iter()
                .map(|v| {
                    Some(ExprOrSpread {
                        spread: None,
                        expr: Box::new(value_to_expr(v)),
                    })
                })
                .collect();

            Expr::Array(ArrayLit {
                span: Default::default(),
                elems,
            })
        }

        JsValue::Object(obj) => {
            let props: Vec<PropOrSpread> = obj
                .borrow()
                .iter()
                .map(|(k, v)| {
                    PropOrSpread::Prop(Box::new(Prop::KeyValue(KeyValueProp {
                        key: PropName::Ident(IdentName {
                            span: Default::default(),
                            sym: k.clone().into(),
                        }),
                        value: Box::new(value_to_expr(v)),
                    })))
                })
                .collect();

            Expr::Object(ObjectLit {
                span: Default::default(),
                props,
            })
        }

        JsValue::Function(fv) => {
            match fv {
                FunctionValue::Known { params, body } => {
                    let fn_params: Vec<Param> = params
                        .iter()
                        .map(|p| Param {
                            span: Default::default(),
                            decorators: vec![],
                            pat: Pat::Ident(BindingIdent {
                                id: Ident {
                                    span: Default::default(),
                                    ctxt: Default::default(),
                                    sym: p.clone().into(),
                                    optional: false,
                                },
                                type_ann: None,
                            }),
                        })
                        .collect();

                    Expr::Fn(FnExpr {
                        ident: None,
                        function: Box::new(Function {
                            params: fn_params,
                            decorators: vec![],
                            span: Default::default(),
                            ctxt: Default::default(),
                            body: Some(body.clone()),
                            is_generator: false,
                            is_async: false,
                            type_params: None,
                            return_type: None,
                        }),
                    })
                }
                FunctionValue::DispatchHandler(var_name, n) => {
                    // Reference to var_name[n]
                    Expr::Member(MemberExpr {
                        span: Default::default(),
                        obj: Box::new(Expr::Ident(Ident {
                            span: Default::default(),
                            ctxt: Default::default(),
                            sym: var_name.clone().into(),
                            optional: false,
                        })),
                        prop: MemberProp::Computed(ComputedPropName {
                            span: Default::default(),
                            expr: Box::new(Expr::Lit(Lit::Num(Number {
                                span: Default::default(),
                                value: *n as f64,
                                raw: None,
                            }))),
                        }),
                    })
                }
                FunctionValue::Opaque(name) => {
                    Expr::Ident(Ident {
                        span: Default::default(),
                        ctxt: Default::default(),
                        sym: name.clone().into(),
                        optional: false,
                    })
                }
            }
        }
    }
}

/// Generate a variable declaration statement
pub fn var_decl(name: &str, value: &AbstractValue) -> Stmt {
    Stmt::Decl(Decl::Var(Box::new(VarDecl {
        span: Default::default(),
        ctxt: Default::default(),
        kind: VarDeclKind::Var,
        declare: false,
        decls: vec![VarDeclarator {
            span: Default::default(),
            name: Pat::Ident(BindingIdent {
                id: Ident {
                    span: Default::default(),
                    ctxt: Default::default(),
                    sym: name.into(),
                    optional: false,
                },
                type_ann: None,
            }),
            init: Some(Box::new(value_to_expr(value))),
            definite: false,
        }],
    })))
}

/// Generate an assignment statement
pub fn assign_stmt(name: &str, value: &AbstractValue) -> Stmt {
    Stmt::Expr(ExprStmt {
        span: Default::default(),
        expr: Box::new(Expr::Assign(AssignExpr {
            span: Default::default(),
            op: AssignOp::Assign,
            left: AssignTarget::Simple(SimpleAssignTarget::Ident(BindingIdent {
                id: Ident {
                    span: Default::default(),
                    ctxt: Default::default(),
                    sym: name.into(),
                    optional: false,
                },
                type_ann: None,
            })),
            right: Box::new(value_to_expr(value)),
        })),
    })
}

/// Generate JavaScript source code from a module
pub fn emit_module(module: &Module) -> String {
    use swc_common::{sync::Lrc, SourceMap};
    use swc_ecma_codegen::{text_writer::JsWriter, Emitter, Config};

    let cm: Lrc<SourceMap> = Default::default();
    let mut buf = vec![];

    {
        let writer = JsWriter::new(cm.clone(), "\n", &mut buf, None);
        let mut emitter = Emitter {
            cfg: Config::default(),
            cm: cm.clone(),
            comments: None,
            wr: writer,
        };

        emitter.emit_module(module).unwrap();
    }

    String::from_utf8(buf).unwrap()
}

/// Generate JavaScript source code from a list of statements
pub fn emit_stmts(stmts: &[Stmt]) -> String {
    let module = Module {
        span: Default::default(),
        shebang: None,
        body: stmts
            .iter()
            .map(|s| ModuleItem::Stmt(s.clone()))
            .collect(),
    };

    emit_module(&module)
}

/// Create an identifier expression
pub fn ident(name: &str) -> Expr {
    Expr::Ident(Ident {
        span: Default::default(),
        ctxt: Default::default(),
        sym: name.into(),
        optional: false,
    })
}

/// Create a number literal expression
pub fn num(n: f64) -> Expr {
    Expr::Lit(Lit::Num(Number {
        span: Default::default(),
        value: n,
        raw: None,
    }))
}

/// Create a binary expression
pub fn binop(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Bin(BinExpr {
        span: Default::default(),
        op,
        left: Box::new(left),
        right: Box::new(right),
    })
}

/// Create a call expression
pub fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    Expr::Call(CallExpr {
        span: Default::default(),
        ctxt: Default::default(),
        callee: Callee::Expr(Box::new(callee)),
        args: args
            .into_iter()
            .map(|e| ExprOrSpread {
                spread: None,
                expr: Box::new(e),
            })
            .collect(),
        type_args: None,
    })
}

/// Create a unary expression
pub fn unary(op: UnaryOp, arg: Expr) -> Expr {
    Expr::Unary(UnaryExpr {
        span: Default::default(),
        op,
        arg: Box::new(arg),
    })
}

/// Create a method call expression: obj.method(args)
pub fn method_call(obj: Expr, method: &str, args: Vec<Expr>) -> Expr {
    Expr::Call(CallExpr {
        span: Default::default(),
        ctxt: Default::default(),
        callee: Callee::Expr(Box::new(Expr::Member(MemberExpr {
            span: Default::default(),
            obj: Box::new(obj),
            prop: MemberProp::Ident(IdentName {
                span: Default::default(),
                sym: method.into(),
            }),
        }))),
        args: args
            .into_iter()
            .map(|e| ExprOrSpread {
                spread: None,
                expr: Box::new(e),
            })
            .collect(),
        type_args: None,
    })
}

/// Create a member access expression: obj.prop
pub fn member(obj: Expr, prop: &str) -> Expr {
    Expr::Member(MemberExpr {
        span: Default::default(),
        obj: Box::new(obj),
        prop: MemberProp::Ident(IdentName {
            span: Default::default(),
            sym: prop.into(),
        }),
    })
}

/// Create a member expression statement
pub fn expr_stmt(expr: Expr) -> Stmt {
    Stmt::Expr(ExprStmt {
        span: Default::default(),
        expr: Box::new(expr),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_number() {
        let val = AbstractValue::known_number(42.0);
        let expr = value_to_expr(&val);
        // Just verify it doesn't panic
        assert!(matches!(expr, Expr::Lit(Lit::Num(_))));
    }

    #[test]
    fn test_emit_array() {
        let val = AbstractValue::known_array(vec![
            AbstractValue::known_number(1.0),
            AbstractValue::known_number(2.0),
        ]);
        let expr = value_to_expr(&val);
        assert!(matches!(expr, Expr::Array(_)));
    }

    #[test]
    fn test_emit_stmts() {
        let stmts = vec![
            var_decl("x", &AbstractValue::known_number(42.0)),
        ];
        let code = emit_stmts(&stmts);
        assert!(code.contains("var x = 42"));
    }
}
