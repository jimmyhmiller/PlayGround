//! M1 type/convention checks.
//!
//! Everything is i64, so "type checking" here is: variables are bound, calls
//! refer to defined functions with matching arity, and every function's
//! convention exists and has a lowering we can actually emit. The last check is
//! where the design's "calling convention is part of the type" shows up even in
//! this tiny slice: a function whose convention can't be lowered is a *compile
//! error*, not silent miscompilation.

use std::collections::HashSet;

use crate::ast::*;

pub fn check(program: &Program) -> Result<(), String> {
    // arity table for call checking
    let arity: std::collections::HashMap<&str, usize> = program
        .funcs
        .iter()
        .map(|f| (f.name.as_str(), f.params.len()))
        .collect();

    for f in &program.funcs {
        // convention must exist and be lowerable
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("function '{}': unknown convention '{}'", f.name, f.cc))?;
        if conv.native_id().is_none() {
            return Err(format!(
                "function '{}': convention '{}' has no native lowering \
                 (shim/trampoline path is M2, not implemented yet)",
                f.name, f.cc
            ));
        }

        let mut scope: HashSet<String> = f.params.iter().map(|p| p.name.clone()).collect();
        for e in &f.body {
            check_expr(e, &mut scope, &arity, &f.name)?;
        }
    }
    Ok(())
}

fn check_expr(
    e: &Expr,
    scope: &mut HashSet<String>,
    arity: &std::collections::HashMap<&str, usize>,
    fname: &str,
) -> Result<(), String> {
    match e {
        Expr::Int(_) => Ok(()),
        Expr::Var(name) => {
            if scope.contains(name) {
                Ok(())
            } else {
                Err(format!("in '{fname}': unbound variable '{name}'"))
            }
        }
        Expr::Bin { lhs, rhs, .. } | Expr::Cmp { lhs, rhs, .. } => {
            check_expr(lhs, scope, arity, fname)?;
            check_expr(rhs, scope, arity, fname)
        }
        Expr::If { cond, then, els } => {
            check_expr(cond, scope, arity, fname)?;
            check_expr(then, scope, arity, fname)?;
            check_expr(els, scope, arity, fname)
        }
        Expr::Do(es) => {
            for e in es {
                check_expr(e, scope, arity, fname)?;
            }
            Ok(())
        }
        Expr::Let { binds, body } => {
            // lexical: bindings are sequential, body sees them, outer scope is
            // restored afterward.
            let mut child = scope.clone();
            for (name, val) in binds {
                check_expr(val, &mut child, arity, fname)?;
                child.insert(name.clone());
            }
            for e in body {
                check_expr(e, &mut child, arity, fname)?;
            }
            Ok(())
        }
        Expr::Call { func, args } => {
            match arity.get(func.as_str()) {
                None => return Err(format!("in '{fname}': call to undefined function '{func}'")),
                Some(n) if *n != args.len() => {
                    return Err(format!(
                        "in '{fname}': '{func}' expects {n} args, got {}",
                        args.len()
                    ))
                }
                Some(_) => {}
            }
            for a in args {
                check_expr(a, scope, arity, fname)?;
            }
            Ok(())
        }
    }
}
