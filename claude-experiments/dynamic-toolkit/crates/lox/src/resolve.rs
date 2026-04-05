//! Resolve variable captures for closures.
//!
//! Walks the AST and determines, for each function:
//! - Which of its locals are captured by child functions (must be upvalue cells)
//! - Which variables it captures from enclosing scopes (its upvalue list)

use std::collections::{HashMap, HashSet};

use crate::ast::*;

/// Capture info for a single function.
#[derive(Debug, Default)]
pub struct FunCaptures {
    /// Upvalue names this function needs, in order. Each is a variable
    /// from an enclosing scope.
    pub upvalues: Vec<String>,
    /// Locals of this function that are captured by a child and must
    /// live in upvalue cells instead of plain stack slots.
    pub captured_locals: HashSet<String>,
    /// Is this a method (has implicit `this` parameter)?
    pub is_method: bool,
}

/// Result of the resolution pass.
pub struct ResolveResult {
    /// Capture info for the top-level script.
    pub script: FunCaptures,
    /// Capture info keyed by function name.
    /// For methods: keyed by "ClassName.methodName".
    pub functions: HashMap<String, FunCaptures>,
}

pub fn resolve(program: &Program) -> ResolveResult {
    let mut r = Resolver {
        scopes: vec![],
        result: ResolveResult {
            script: FunCaptures::default(),
            functions: HashMap::new(),
        },
    };
    // Script scope
    r.push_scope("__script__", false);
    for stmt in &program.stmts {
        r.resolve_stmt(stmt);
    }
    r.result.script = r.pop_scope();
    r.result
}

struct Resolver {
    scopes: Vec<Scope>,
    result: ResolveResult,
}

struct Scope {
    name: String,
    is_method: bool,
    locals: HashSet<String>,
    /// Upvalues this scope needs (name + where it comes from).
    upvalues: Vec<String>,
    upvalue_set: HashSet<String>,
    /// This scope's locals that are captured by children.
    captured_locals: HashSet<String>,
}

impl Resolver {
    fn push_scope(&mut self, name: &str, is_method: bool) {
        self.scopes.push(Scope {
            name: name.to_string(),
            is_method,
            locals: HashSet::new(),
            upvalues: Vec::new(),
            upvalue_set: HashSet::new(),
            captured_locals: HashSet::new(),
        });
    }

    fn pop_scope(&mut self) -> FunCaptures {
        let scope = self.scopes.pop().unwrap();
        FunCaptures {
            upvalues: scope.upvalues,
            captured_locals: scope.captured_locals,
            is_method: scope.is_method,
        }
    }

    fn declare_local(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.locals.insert(name.to_string());
        }
    }

    fn resolve_var(&mut self, name: &str) {
        if name == "this" || name == "super" {
            // this/super are handled specially — treated like locals in method scopes
        }

        let n = self.scopes.len();
        if n == 0 {
            return;
        }

        // Search from innermost scope outward
        // If found in a scope other than the current one, it's a capture.
        for i in (0..n).rev() {
            if self.scopes[i].locals.contains(name) {
                if i == n - 1 {
                    // Found in current scope — just a local, nothing to do
                    return;
                }
                // Found in outer scope i. Mark it captured there,
                // and thread it as an upvalue through all intermediate scopes.
                self.scopes[i].captured_locals.insert(name.to_string());
                for j in (i + 1)..n {
                    if !self.scopes[j].upvalue_set.contains(name) {
                        self.scopes[j].upvalues.push(name.to_string());
                        self.scopes[j].upvalue_set.insert(name.to_string());
                    }
                }
                return;
            }
            // Also check upvalues (for threading through multiple levels)
            if self.scopes[i].upvalue_set.contains(name) {
                if i == n - 1 {
                    return; // Already an upvalue in the current scope
                }
                for j in (i + 1)..n {
                    if !self.scopes[j].upvalue_set.contains(name) {
                        self.scopes[j].upvalues.push(name.to_string());
                        self.scopes[j].upvalue_set.insert(name.to_string());
                    }
                }
                return;
            }
        }
        // Not found in any scope — must be a global. Nothing to do.
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(e) | Stmt::Print(e) => self.resolve_expr(e),
            Stmt::Var(name, init) => {
                if let Some(e) = init {
                    self.resolve_expr(e);
                }
                self.declare_local(name);
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.resolve_stmt(s);
                }
            }
            Stmt::If(cond, then_br, else_br) => {
                self.resolve_expr(cond);
                self.resolve_stmt(then_br);
                if let Some(e) = else_br {
                    self.resolve_stmt(e);
                }
            }
            Stmt::While(cond, body) => {
                self.resolve_expr(cond);
                self.resolve_stmt(body);
            }
            Stmt::Return(e) => {
                if let Some(e) = e {
                    self.resolve_expr(e);
                }
            }
            Stmt::Fun(decl) => {
                // The function name is a local in the enclosing scope
                self.declare_local(&decl.name);
                self.resolve_fun(decl, false);
            }
            Stmt::Class(decl) => {
                self.declare_local(&decl.name);
                if let Some(super_name) = &decl.superclass {
                    self.resolve_var(super_name);
                    // Push a scope for 'super' around the methods
                    self.push_scope(&format!("{}__super", decl.name), false);
                    self.declare_local("super");
                    for method in &decl.methods {
                        let key = format!("{}.{}", decl.name, method.name);
                        self.resolve_fun_with_key(method, true, &key);
                    }
                    let super_captures = self.pop_scope();
                    // Store super scope captures (not used directly but threads upvalues)
                    self.result.functions.insert(format!("{}__super", decl.name), super_captures);
                } else {
                    for method in &decl.methods {
                        let key = format!("{}.{}", decl.name, method.name);
                        self.resolve_fun_with_key(method, true, &key);
                    }
                }
            }
        }
    }

    fn resolve_fun(&mut self, decl: &FunDecl, is_method: bool) {
        self.resolve_fun_with_key(decl, is_method, &decl.name);
    }

    fn resolve_fun_with_key(&mut self, decl: &FunDecl, is_method: bool, key: &str) {
        self.push_scope(key, is_method);
        if is_method {
            self.declare_local("this");
        }
        for p in &decl.params {
            self.declare_local(p);
        }
        for stmt in &decl.body {
            self.resolve_stmt(stmt);
        }
        let captures = self.pop_scope();
        self.result.functions.insert(key.to_string(), captures);
    }

    fn resolve_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Var(name) => self.resolve_var(name),
            Expr::Assign(name, val) => {
                self.resolve_expr(val);
                self.resolve_var(name);
            }
            Expr::Binary(l, _, r) | Expr::Logical(l, _, r) => {
                self.resolve_expr(l);
                self.resolve_expr(r);
            }
            Expr::Unary(_, e) | Expr::Grouping(e) => self.resolve_expr(e),
            Expr::Call(callee, args) => {
                self.resolve_expr(callee);
                for a in args {
                    self.resolve_expr(a);
                }
            }
            Expr::Get(obj, _) => self.resolve_expr(obj),
            Expr::Set(obj, _, val) => {
                self.resolve_expr(obj);
                self.resolve_expr(val);
            }
            Expr::This => self.resolve_var("this"),
            Expr::Super(_) => {
                self.resolve_var("this");
                self.resolve_var("super");
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Nil => {}
        }
    }
}
