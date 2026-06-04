//! Lower the supported subset of real JavaScript (an SWC AST) to the
//! `partial::js` AST (`FuncDef`/`Stmt`/`Expr` with integer-slot locals).
//!
//! Supported: numbers (integer i64), strings, booleans, `undefined`; objects
//! and arrays; member (`o.k`) and index (`a[i]`) read and write; `+ - *` and the
//! comparisons `< <= > >=` plus `=== !==`; unary `-`; `if`/`else`, `while`,
//! `for` (desugared), `switch`/`break`, `return`; named function declarations
//! and **non-capturing** arrow / function expressions (lifted to top level);
//! `arr.push(x)` as a statement.
//!
//! Everything else is a hard error (e.g. closures capturing outer variables,
//! `for..of`/`for..in`, `&&`/`||`, `/`, `%`, classes, `try`, regex, `null`,
//! non-integer numbers, destructuring). No silent fallbacks.

use std::collections::{HashMap, HashSet};

use partial::js::{Bop, Clause, Expr, FuncDef, Stmt};
use swc_ecma_ast as ast;

type R<T> = Result<T, String>;

/// Compile JS source to a list of `FuncDef`s (fid = index). Requires a
/// `function main(input) { ... }` entry.
pub fn compile(src: &str) -> R<Vec<FuncDef>> {
    let module = crate::parse(src)?;
    let mut c = Compiler::default();
    let (module_body, explicit_main) = c.collect_fns(&module)?;
    if explicit_main {
        if !module_body.is_empty() {
            return Err("top-level statements alongside an explicit `function main` \
                        are not supported; put the code inside `main`"
                .into());
        }
    } else if !module_body.is_empty() {
        // Synthesize `main` from the program's top-level statements.
        let fid = c.decls.len();
        c.name_to_fid.insert("main".to_string(), fid);
        c.decls.push(DeclFn { name: "main".to_string(), function: None, module_body: Some(module_body) });
    } else {
        return Err("no `function main(input)` and no top-level code to specialize".into());
    }
    c.lower_declared()?;
    Ok(c.finish())
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[derive(Default)]
struct Compiler {
    /// Declared top-level functions, cloned from the module, indexed by fid.
    decls: Vec<DeclFn>,
    name_to_fid: HashMap<String, usize>,
    /// Lowered declared functions (fid 0..decls.len()), then lifted functions.
    declared: Vec<FuncDef>,
    lifted: Vec<FuncDef>,
}

struct DeclFn {
    name: String,
    /// A real declared function, or `None` for the synthetic module `main`
    /// (whose body is the program's top-level statements).
    function: Option<ast::Function>,
    module_body: Option<Vec<ast::Stmt>>,
}

/// A bound name's residual slot plus whether it is BOXED (a `{value}` cell, for
/// capture-by-reference).
#[derive(Clone, Copy)]
struct Binding {
    slot: usize,
    boxed: bool,
}

/// Per-function lowering state: a lexical stack of name→binding scopes.
struct FnCtx {
    scopes: Vec<HashMap<String, Binding>>,
    next_slot: usize,
    max_slot: usize,
    /// nesting depth of enclosing `switch`/loop, to validate `break`.
    break_depth: usize,
    /// nesting depth of enclosing *loops*, to validate `continue` (which, unlike
    /// `break`, does not target a `switch`).
    loop_depth: usize,
    /// Names bound in enclosing functions (would-be captures). Referencing one
    /// from a lifted arrow is a clear "capturing closure" error; a name in
    /// neither the local scopes nor here is treated as a runtime global.
    outer_names: HashSet<String>,
    /// Names that must be boxed in THIS function (captured + mutated `var`s or
    /// parameters).
    boxed_set: HashSet<String>,
    /// Of `boxed_set`, the names that are PARAMETERS. Their `{value}` cell is
    /// initialized from the incoming argument (`p = {value: p}`), not from
    /// `undefined` like a hoisted local.
    boxed_params: HashSet<String>,
    /// Names of nested `function` declarations hoisted to the top of THIS
    /// function body (so `lower_stmt` skips them at their textual position; a
    /// `function` declaration anywhere else is rejected).
    hoisted_fns: HashSet<String>,
    /// Slot reserved (lazily, on first reference) for this function's
    /// `arguments` object. The engine fills it with the actual call args on the
    /// inline path.
    arguments_slot: Option<usize>,
}

impl FnCtx {
    fn new(outer_names: HashSet<String>) -> Self {
        FnCtx {
            scopes: vec![HashMap::new()],
            next_slot: 0,
            max_slot: 0,
            break_depth: 0,
            loop_depth: 0,
            outer_names,
            boxed_set: HashSet::new(),
            boxed_params: HashSet::new(),
            hoisted_fns: HashSet::new(),
            arguments_slot: None,
        }
    }
    /// The slot for this function's `arguments`, reserving one on first use.
    fn arguments_slot_or_reserve(&mut self) -> usize {
        if let Some(slot) = self.arguments_slot {
            return slot;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        self.max_slot = self.max_slot.max(self.next_slot);
        self.arguments_slot = Some(slot);
        slot
    }
    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop(&mut self) {
        self.scopes.pop();
    }
    fn declare(&mut self, name: &str) -> usize {
        self.declare_boxed(name, self.boxed_set.contains(name))
    }
    fn declare_boxed(&mut self, name: &str, boxed: bool) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.max_slot = self.max_slot.max(self.next_slot);
        self.scopes.last_mut().unwrap().insert(name.to_string(), Binding { slot, boxed });
        slot
    }
    /// Declare (or reuse) a `var` in the *function* scope. `var` is
    /// function-scoped and hoisted, so it survives inner blocks and a
    /// redeclaration refers to the same binding.
    fn declare_var(&mut self, name: &str) -> usize {
        if let Some(b) = self.scopes[0].get(name) {
            return b.slot;
        }
        let boxed = self.boxed_set.contains(name);
        let slot = self.next_slot;
        self.next_slot += 1;
        self.max_slot = self.max_slot.max(self.next_slot);
        self.scopes[0].insert(name.to_string(), Binding { slot, boxed });
        slot
    }
    fn resolve(&self, name: &str) -> Option<usize> {
        self.scopes.iter().rev().find_map(|s| s.get(name).map(|b| b.slot))
    }
    fn resolve_binding(&self, name: &str) -> Option<Binding> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }
    /// All names visible here (local scopes + already-inherited outer names),
    /// i.e. the would-be captures for an arrow lifted at this point.
    fn visible_names(&self) -> HashSet<String> {
        let mut names = self.outer_names.clone();
        for scope in &self.scopes {
            names.extend(scope.keys().cloned());
        }
        names
    }
}

impl Compiler {
    /// Register top-level function declarations and gather the remaining
    /// top-level statements (the module body: `var`s, IIFEs, expression
    /// statements) into `module_body` for a synthetic `main`.
    fn collect_fns(&mut self, module: &ast::Module) -> R<(Vec<ast::Stmt>, bool)> {
        let mut module_body: Vec<ast::Stmt> = Vec::new();
        let mut explicit_main = false;
        for item in &module.body {
            match item {
                ast::ModuleItem::Stmt(ast::Stmt::Decl(ast::Decl::Fn(f))) => {
                    let name = f.ident.sym.to_string();
                    if self.name_to_fid.contains_key(&name) {
                        return Err(format!("duplicate function `{name}`"));
                    }
                    if name == "main" {
                        explicit_main = true;
                    }
                    let fid = self.decls.len();
                    self.name_to_fid.insert(name.clone(), fid);
                    self.decls.push(DeclFn {
                        name,
                        function: Some((*f.function).clone()),
                        module_body: None,
                    });
                }
                ast::ModuleItem::Stmt(ast::Stmt::Empty(_)) => {}
                ast::ModuleItem::Stmt(s) => module_body.push(s.clone()),
                ast::ModuleItem::ModuleDecl(_) => {
                    return Err("ES module imports/exports are not supported".into());
                }
            }
        }
        Ok((module_body, explicit_main))
    }

    fn lower_declared(&mut self) -> R<()> {
        for i in 0..self.decls.len() {
            let name = self.decls[i].name.clone();
            let (nslots, nparams, arguments_slot, body) = match self.decls[i].function.clone() {
                Some(func) => self.lower_fn(&func)?,
                None => {
                    let stmts = self.decls[i].module_body.clone().unwrap();
                    self.lower_module_main(&stmts)?
                }
            };
            self.declared.push(FuncDef {
                name: leak(name),
                nslots,
                ncaptured: 0,
                nparams,
                slot_names: vec![],
                arguments_slot,
                body,
            });
        }
        Ok(())
    }

    /// Lower the program's top-level statements as the body of a synthetic
    /// `main(input)` (the engine entry). `input` is the one dynamic value; the
    /// top-level `var`s and code run in order.
    fn lower_module_main(&mut self, stmts: &[ast::Stmt]) -> R<(usize, usize, Option<usize>, Vec<Stmt>)> {
        let mut ctx = FnCtx::new(HashSet::new());
        ctx.boxed_set = boxed_locals(&["input"], stmts);
        if ctx.boxed_set.contains("input") {
            ctx.boxed_params.insert("input".to_string());
        }
        ctx.declare("input");
        hoist_vars(stmts, &mut ctx);
        self.hoist_fn_decls(stmts, &mut ctx);
        let mut body = self.boxed_cell_stmts(&ctx);
        body.extend(self.lower_fn_decls(stmts, &mut ctx)?);
        body.extend(self.lower_stmts(stmts, &mut ctx)?);
        Ok((ctx.max_slot.max(1), 1, ctx.arguments_slot, body))
    }

    fn finish(mut self) -> Vec<FuncDef> {
        self.declared.append(&mut self.lifted);
        self.declared
    }

    /// Create the `{value: undefined}` cells for this function's own boxed vars,
    /// at the top of its body (so capture-by-reference cells exist before use).
    fn boxed_cell_stmts(&self, ctx: &FnCtx) -> Vec<Stmt> {
        let mut names: Vec<&String> = ctx.boxed_set.iter().collect();
        names.sort();
        names
            .into_iter()
            .filter_map(|name| ctx.resolve(name).map(|slot| (name, slot)))
            .map(|(name, slot)| {
                // A boxed parameter starts holding its incoming argument; wrap it
                // in place (`p = {value: p}`, the inner `Var(slot)` reading the raw
                // arg before the assignment rebinds the slot). A boxed local starts
                // `undefined` (it is hoisted).
                let init = if ctx.boxed_params.contains(name) {
                    Expr::Var(slot)
                } else {
                    Expr::Undefined
                };
                Stmt::Let(slot, Expr::Object(vec![("value".to_string(), init)]))
            })
            .collect()
    }

    /// Declare a slot (function-scoped, hoisted) for each direct-child
    /// `function` declaration of `stmts`, and record the names so `lower_stmt`
    /// skips them at their textual position. Must run after `hoist_vars` and
    /// after `ctx.boxed_set` is set (so the slot picks up the boxed flag).
    fn hoist_fn_decls(&self, stmts: &[ast::Stmt], ctx: &mut FnCtx) {
        for name in fn_decl_names(stmts) {
            ctx.declare_var(&name);
            ctx.hoisted_fns.insert(name);
        }
    }

    /// Lower the direct-child `function` declarations of `stmts` into assignment
    /// statements, emitted at the top of the body (JS function-declaration
    /// hoisting). Each becomes a closure assigned to its hoisted slot (or its
    /// `{value}` cell when boxed for capture-by-reference).
    fn lower_fn_decls(&mut self, stmts: &[ast::Stmt], ctx: &mut FnCtx) -> R<Vec<Stmt>> {
        let mut out = Vec::new();
        for s in stmts {
            let ast::Stmt::Decl(ast::Decl::Fn(f)) = s else {
                continue;
            };
            let name = f.ident.sym.to_string();
            let params: Vec<ast::Pat> =
                f.function.params.iter().map(|p| p.pat.clone()).collect();
            let body = match &f.function.body {
                Some(b) => b.stmts.clone(),
                None => return Err(format!("function `{name}` without a body")),
            };
            let (fid, captures) =
                self.lift_arrow(&params, ArrowBody::BlockOwned(body), ctx)?;
            let caps = self.capture_exprs(&captures, ctx)?;
            let closure = Expr::Closure(fid, caps);
            let b = ctx
                .resolve_binding(&name)
                .expect("fn-decl slot declared by hoist_fn_decls");
            if b.boxed {
                out.push(Stmt::SetProp(Expr::Var(b.slot), "value".to_string(), closure));
            } else {
                out.push(Stmt::Set(b.slot, closure));
            }
        }
        Ok(out)
    }

    /// Lower a function (declaration or lifted expression). Returns
    /// (nslots, nparams, arguments_slot, body).
    fn lower_fn(&mut self, function: &ast::Function) -> R<(usize, usize, Option<usize>, Vec<Stmt>)> {
        let block = match &function.body {
            Some(b) => b,
            None => return Err("function without a body".into()),
        };
        let mut ctx = FnCtx::new(HashSet::new());
        let param_names: Vec<&str> =
            function.params.iter().map(|p| pat_ident(&p.pat)).collect::<R<Vec<_>>>()?;
        ctx.boxed_set = boxed_locals(&param_names, &block.stmts);
        for name in &param_names {
            if ctx.boxed_set.contains(*name) {
                ctx.boxed_params.insert(name.to_string());
            }
            ctx.declare(name);
        }
        let nparams = function.params.len();
        hoist_vars(&block.stmts, &mut ctx);
        self.hoist_fn_decls(&block.stmts, &mut ctx);
        let mut body = self.boxed_cell_stmts(&ctx);
        body.extend(self.lower_fn_decls(&block.stmts, &mut ctx)?);
        body.extend(self.lower_stmts(&block.stmts, &mut ctx)?);
        Ok((ctx.max_slot.max(nparams), nparams, ctx.arguments_slot, body))
    }

    /// Lift an arrow / function expression to a top-level function via closure
    /// conversion: its free variables that are visible in the enclosing scope
    /// become leading *captured* parameters (slots `0..ncaptured`). Returns the
    /// new fid and the captured names (in capture-slot order) so the caller emits
    /// `Closure(fid, [capture values])`.
    ///
    /// Captures are by VALUE (snapshot at closure creation). This matches JS for
    /// immediately-invoked functions and for captures that aren't reassigned
    /// after creation; a long-lived closure observing a later mutation of a
    /// captured variable is NOT modeled (Node validation would surface it).
    fn lift_arrow(
        &mut self,
        params: &[ast::Pat],
        body: ArrowBody,
        enclosing: &FnCtx,
    ) -> R<(usize, Vec<(String, bool)>)> {
        // free variables visible outside become captures (deterministic order),
        // each tagged with whether it is a boxed cell in the enclosing scope.
        let free = match &body {
            ArrowBody::Block(s) => free_vars(params, s),
            ArrowBody::BlockOwned(s) => free_vars(params, s),
            ArrowBody::Expr(e) => {
                let mut used = HashSet::new();
                used_in_expr(e, &mut used);
                let mut bound = HashSet::new();
                for p in params {
                    bound_in_pat(p, &mut bound);
                }
                let mut f: Vec<String> = used.into_iter().filter(|n| !bound.contains(n)).collect();
                f.sort();
                f
            }
        };
        let visible = enclosing.visible_names();
        let captures: Vec<(String, bool)> = free
            .into_iter()
            .filter(|n| visible.contains(n))
            .map(|n| {
                let boxed = enclosing.resolve_binding(&n).map_or(false, |b| b.boxed);
                (n, boxed)
            })
            .collect();

        let own_body: Vec<ast::Stmt> = match &body {
            ArrowBody::Block(s) => s.to_vec(),
            ArrowBody::BlockOwned(s) => s.clone(),
            ArrowBody::Expr(_) => Vec::new(),
        };
        let mut ctx = FnCtx::new(HashSet::new());
        let param_names: Vec<&str> = params.iter().map(pat_ident).collect::<R<Vec<_>>>()?;
        ctx.boxed_set = boxed_locals(&param_names, &own_body);
        for (name, boxed) in &captures {
            ctx.declare_boxed(name, *boxed); // captured slots first; cell from enclosing
        }
        for name in &param_names {
            if ctx.boxed_set.contains(*name) {
                ctx.boxed_params.insert(name.to_string());
            }
            ctx.declare(name);
        }
        let ncaptured = captures.len();
        let nparams = params.len();
        let stmts = match body {
            ArrowBody::Block(stmts) => {
                hoist_vars(stmts, &mut ctx);
                self.hoist_fn_decls(stmts, &mut ctx);
                let mut b = self.boxed_cell_stmts(&ctx);
                b.extend(self.lower_fn_decls(stmts, &mut ctx)?);
                b.extend(self.lower_stmts(stmts, &mut ctx)?);
                b
            }
            ArrowBody::BlockOwned(stmts) => {
                hoist_vars(&stmts, &mut ctx);
                self.hoist_fn_decls(&stmts, &mut ctx);
                let mut b = self.boxed_cell_stmts(&ctx);
                b.extend(self.lower_fn_decls(&stmts, &mut ctx)?);
                b.extend(self.lower_stmts(&stmts, &mut ctx)?);
                b
            }
            ArrowBody::Expr(e) => {
                let r = self.lower_expr(e, &mut ctx)?;
                vec![Stmt::Return(r)]
            }
        };
        let fid = self.decls.len() + self.lifted.len();
        self.lifted.push(FuncDef {
            name: leak(format!("lambda#{fid}")),
            nslots: ctx.max_slot.max(ncaptured + nparams),
            ncaptured,
            nparams,
            slot_names: vec![],
            arguments_slot: ctx.arguments_slot,
            body: stmts,
        });
        Ok((fid, captures))
    }

    /// Lower the capture values for a lifted closure: the enclosing slot's
    /// content (the cell itself if boxed, else the value).
    fn capture_exprs(&self, captures: &[(String, bool)], ctx: &FnCtx) -> R<Vec<Expr>> {
        let mut caps = Vec::with_capacity(captures.len());
        for (name, _boxed) in captures {
            let slot = ctx
                .resolve(name)
                .ok_or_else(|| format!("capture of unbound `{name}`"))?;
            caps.push(Expr::Var(slot));
        }
        Ok(caps)
    }

    // ---- statements ----

    fn lower_stmts(&mut self, stmts: &[ast::Stmt], ctx: &mut FnCtx) -> R<Vec<Stmt>> {
        let mut out = Vec::new();
        for s in stmts {
            self.lower_stmt(s, ctx, &mut out)?;
        }
        Ok(out)
    }

    /// Lower a block or single statement into a fresh `Vec<Stmt>` (its own scope).
    fn lower_body(&mut self, s: &ast::Stmt, ctx: &mut FnCtx) -> R<Vec<Stmt>> {
        ctx.push();
        let mut out = Vec::new();
        match s {
            ast::Stmt::Block(b) => {
                for st in &b.stmts {
                    self.lower_stmt(st, ctx, &mut out)?;
                }
            }
            other => self.lower_stmt(other, ctx, &mut out)?,
        }
        ctx.pop();
        Ok(out)
    }

    fn lower_stmt(&mut self, s: &ast::Stmt, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        match s {
            ast::Stmt::Decl(ast::Decl::Var(v)) => self.lower_var_decl(v, ctx, out)?,
            ast::Stmt::Decl(ast::Decl::Fn(f)) => {
                // Direct-child function declarations were hoisted+assigned at the
                // top of the body (see lower_fn_decls); skip them here. A
                // function declaration nested inside a block is not hoisted by
                // this subset.
                let name = f.ident.sym.to_string();
                if !ctx.hoisted_fns.contains(&name) {
                    return Err(format!(
                        "`function {name}` declared inside a block is not supported; \
                         move it to the top level of the function body"
                    ));
                }
            }
            ast::Stmt::Decl(_) => return Err("unsupported declaration (class/using/etc.)".into()),
            ast::Stmt::Expr(e) => self.lower_expr_stmt(&e.expr, ctx, out)?,
            ast::Stmt::Return(r) => {
                let e = match &r.arg {
                    Some(e) => self.lower_expr(e, ctx)?,
                    None => Expr::Undefined,
                };
                out.push(Stmt::Return(e));
            }
            ast::Stmt::If(i) => {
                let cond = self.lower_expr(&i.test, ctx)?;
                let cons = self.lower_body(&i.cons, ctx)?;
                let alt = match &i.alt {
                    Some(a) => self.lower_body(a, ctx)?,
                    None => vec![],
                };
                out.push(Stmt::If(cond, cons, alt));
            }
            ast::Stmt::While(w) => {
                let cond = self.lower_expr(&w.test, ctx)?;
                ctx.break_depth += 1;
                ctx.loop_depth += 1;
                let body = self.lower_body(&w.body, ctx)?;
                ctx.loop_depth -= 1;
                ctx.break_depth -= 1;
                out.push(Stmt::While(cond, body));
            }
            ast::Stmt::For(f) => self.lower_for(f, ctx, out)?,
            ast::Stmt::Switch(sw) => self.lower_switch(sw, ctx, out)?,
            ast::Stmt::Break(b) => {
                if b.label.is_some() {
                    return Err("labeled `break` is not supported".into());
                }
                if ctx.break_depth == 0 {
                    return Err("`break` outside of a switch or loop".into());
                }
                out.push(Stmt::Break);
            }
            ast::Stmt::Continue(c) => {
                if c.label.is_some() {
                    return Err("labeled `continue` is not supported".into());
                }
                if ctx.loop_depth == 0 {
                    return Err("`continue` outside of a loop".into());
                }
                out.push(Stmt::Continue);
            }
            ast::Stmt::Block(b) => {
                ctx.push();
                for st in &b.stmts {
                    self.lower_stmt(st, ctx, out)?;
                }
                ctx.pop();
            }
            ast::Stmt::Empty(_) => {}
            ast::Stmt::DoWhile(_) => return Err("`do...while` is not supported".into()),
            ast::Stmt::ForIn(_) | ast::Stmt::ForOf(_) => {
                return Err("`for..in` / `for..of` are not supported (use an index `for`)".into());
            }
            ast::Stmt::Throw(t) => {
                let e = self.lower_expr(&t.arg, ctx)?;
                out.push(Stmt::Throw(e));
            }
            ast::Stmt::Try(t) => self.lower_try(t, ctx, out)?,
            other => return Err(format!("unsupported statement: {:?}", std::mem::discriminant(other))),
        }
        Ok(())
    }

    fn lower_for(&mut self, f: &ast::ForStmt, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        // `for (init; cond; update) body` becomes `init; For { cond, body, update }`.
        // Keeping `update` separate lets `continue` run it before re-testing.
        ctx.push();
        match &f.init {
            Some(ast::VarDeclOrExpr::VarDecl(v)) => self.lower_var_decl(v, ctx, out)?,
            Some(ast::VarDeclOrExpr::Expr(e)) => self.lower_expr_stmt(e, ctx, out)?,
            None => {}
        }
        // An absent test means `true`: `for (;;)` is an infinite loop, exited via
        // `break`/`return`.
        let cond = match &f.test {
            Some(e) => self.lower_expr(e, ctx)?,
            None => Expr::Bool(true),
        };
        ctx.break_depth += 1;
        ctx.loop_depth += 1;
        let body = self.lower_body(&f.body, ctx)?;
        let mut update = Vec::new();
        if let Some(u) = &f.update {
            self.lower_expr_stmt(u, ctx, &mut update)?;
        }
        ctx.loop_depth -= 1;
        ctx.break_depth -= 1;
        out.push(Stmt::For { cond, body, update });
        ctx.pop();
        Ok(())
    }

    fn lower_switch(&mut self, sw: &ast::SwitchStmt, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        let disc = self.lower_expr(&sw.discriminant, ctx)?;
        ctx.break_depth += 1;
        let mut clauses = Vec::new();
        for case in &sw.cases {
            let body = {
                ctx.push();
                let mut b = Vec::new();
                for st in &case.cons {
                    self.lower_stmt(st, ctx, &mut b)?;
                }
                ctx.pop();
                b
            };
            match &case.test {
                Some(t) => {
                    let test = self.lower_expr(t, ctx)?;
                    clauses.push(Clause::Case(test, body));
                }
                None => clauses.push(Clause::Default(body)),
            }
        }
        ctx.break_depth -= 1;
        out.push(Stmt::Switch(disc, clauses));
        Ok(())
    }

    /// Lower a `var`/`let`/`const` declaration (one or more declarators). `var`
    /// is function-scoped and hoisted; `let`/`const` are block-scoped.
    fn lower_var_decl(&mut self, v: &ast::VarDecl, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        let is_var = matches!(v.kind, ast::VarDeclKind::Var);
        for d in &v.decls {
            let name = pat_ident(&d.name)?;
            let init = match &d.init {
                Some(e) => Some(self.lower_expr(e, ctx)?),
                None => None,
            };
            if is_var {
                // Hoisted: the slot already holds `undefined` (or a `{value}` cell
                // if boxed), created at function entry; only an explicit
                // initializer assigns (a bare `var x;` after a write is a no-op).
                let slot = ctx.declare_var(name);
                let boxed = ctx.boxed_set.contains(name);
                if let Some(e) = init {
                    if boxed {
                        out.push(Stmt::SetProp(Expr::Var(slot), "value".to_string(), e));
                    } else {
                        out.push(Stmt::Set(slot, e));
                    }
                }
            } else {
                if ctx.boxed_set.contains(name) {
                    return Err(format!(
                        "`let`/`const` `{name}` is captured and mutated; only `var` \
                         supports capture-by-reference in this subset"
                    ));
                }
                let slot = ctx.declare(name);
                out.push(Stmt::Let(slot, init.unwrap_or(Expr::Undefined)));
            }
        }
        Ok(())
    }

    fn lower_try(&mut self, t: &ast::TryStmt, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        if t.finalizer.is_some() {
            return Err("`finally` is not yet supported".into());
        }
        let handler = match &t.handler {
            Some(h) => h,
            None => return Err("a `try` without `catch` is not supported".into()),
        };
        // try body, in its own scope
        ctx.push();
        let mut body = Vec::new();
        for st in &t.block.stmts {
            self.lower_stmt(st, ctx, &mut body)?;
        }
        ctx.pop();
        // catch body, with the (optional) exception binding in its own scope
        ctx.push();
        let catch_slot = match &handler.param {
            Some(p) => Some(ctx.declare(pat_ident(p)?)),
            None => None,
        };
        let mut catch_body = Vec::new();
        for st in &handler.body.stmts {
            self.lower_stmt(st, ctx, &mut catch_body)?;
        }
        ctx.pop();
        out.push(Stmt::Try { body, catch_slot, catch_body });
        Ok(())
    }

    /// Lower an expression used in statement position (assignments, calls, the
    /// `arr.push(x)` form).
    fn lower_expr_stmt(&mut self, e: &ast::Expr, ctx: &mut FnCtx, out: &mut Vec<Stmt>) -> R<()> {
        match e {
            // `i++` / `--i` etc. in statement position (e.g. a `for` update): the
            // result value is discarded, so pre/post doesn't matter — it's just
            // `target = target ± 1`.
            ast::Expr::Update(u) => {
                let op = match u.op {
                    ast::UpdateOp::PlusPlus => Bop::Add,
                    ast::UpdateOp::MinusMinus => Bop::Sub,
                };
                let one = Box::new(Expr::Num(1));
                // `t++` is `t = ToNumber(t) + 1`, not `t = t + 1` (which would
                // *concatenate* for a string/object `t`: `"3"++` is 4, not "31";
                // `obj++` is NaN, not "[object Object]1"). `x - 0` is ToNumber(x)
                // and folds to nothing when `x` is already numeric. (The `--`/Sub
                // form already coerces, but applying it uniformly is harmless.)
                let to_num = |e: Expr| Expr::Bin(Bop::Sub, Box::new(e), Box::new(Expr::Num(0)));
                match &*u.arg {
                    ast::Expr::Ident(id) => {
                        let name = id.sym.as_str();
                        let b = ctx
                            .resolve_binding(name)
                            .ok_or_else(|| format!("update of unbound variable `{name}`"))?;
                        if b.boxed {
                            let cur = Expr::Get(Box::new(Expr::Var(b.slot)), "value".to_string());
                            out.push(Stmt::SetProp(
                                Expr::Var(b.slot),
                                "value".to_string(),
                                Expr::Bin(op, Box::new(to_num(cur)), one),
                            ));
                        } else {
                            out.push(Stmt::Set(
                                b.slot,
                                Expr::Bin(op, Box::new(to_num(Expr::Var(b.slot))), one),
                            ));
                        }
                    }
                    ast::Expr::Member(m) => {
                        let obj = self.lower_expr(&m.obj, ctx)?;
                        match &m.prop {
                            ast::MemberProp::Ident(id) => {
                                let key = id.sym.to_string();
                                let cur = Expr::Get(Box::new(obj.clone()), key.clone());
                                out.push(Stmt::SetProp(obj, key, Expr::Bin(op, Box::new(to_num(cur)), one)));
                            }
                            ast::MemberProp::Computed(c) => {
                                let idx = self.lower_expr(&c.expr, ctx)?;
                                let cur =
                                    Expr::Index(Box::new(obj.clone()), Box::new(idx.clone()));
                                out.push(Stmt::SetIndex(obj, idx, Expr::Bin(op, Box::new(to_num(cur)), one)));
                            }
                            ast::MemberProp::PrivateName(_) => {
                                return Err("private fields are not supported".into())
                            }
                        }
                    }
                    _ => return Err("`++`/`--` on an unsupported target".into()),
                }
            }
            ast::Expr::Assign(a) => {
                let rhs = self.lower_expr(&a.right, ctx)?;
                // A compound assignment `t <op>= r` desugars to `t = t <op> r`.
                // Modeled arithmetic uses a folding `Bin`; everything else pure
                // (`/`, `%`, bitwise, shifts, `**`) passes through as `Opaque`.
                let op_combine: Option<Combine> = match a.op {
                    ast::AssignOp::Assign => None,
                    ast::AssignOp::AddAssign => Some(Combine::Bin(Bop::Add)),
                    ast::AssignOp::SubAssign => Some(Combine::Bin(Bop::Sub)),
                    ast::AssignOp::MulAssign => Some(Combine::Bin(Bop::Mul)),
                    ast::AssignOp::DivAssign => Some(Combine::Opaque("/")),
                    ast::AssignOp::ModAssign => Some(Combine::Opaque("%")),
                    ast::AssignOp::ExpAssign => Some(Combine::Opaque("**")),
                    // Bitwise / shift compound assignments fold like their binary
                    // forms (modeled `Bop`s), not opaque.
                    ast::AssignOp::BitAndAssign => Some(Combine::Bin(Bop::BitAnd)),
                    ast::AssignOp::BitOrAssign => Some(Combine::Bin(Bop::BitOr)),
                    ast::AssignOp::BitXorAssign => Some(Combine::Bin(Bop::BitXor)),
                    ast::AssignOp::LShiftAssign => Some(Combine::Bin(Bop::Shl)),
                    ast::AssignOp::RShiftAssign => Some(Combine::Bin(Bop::Shr)),
                    ast::AssignOp::ZeroFillRShiftAssign => Some(Combine::Bin(Bop::UShr)),
                    // `&&=` / `||=` / `??=` are short-circuiting (conditional
                    // assignment); their control flow is not modeled here.
                    other => return Err(format!("unsupported assignment operator {:?}", other)),
                };
                // resolve the target
                match &a.left {
                    ast::AssignTarget::Simple(ast::SimpleAssignTarget::Ident(bi)) => {
                        let name = bi.id.sym.as_str();
                        match ctx.resolve_binding(name) {
                            // boxed: write through the `{value}` cell
                            Some(b) if b.boxed => {
                                let value = combine_assign(&op_combine, rhs, || {
                                    Expr::Get(Box::new(Expr::Var(b.slot)), "value".to_string())
                                });
                                out.push(Stmt::SetProp(Expr::Var(b.slot), "value".to_string(), value));
                            }
                            Some(b) => {
                                let value =
                                    combine_assign(&op_combine, rhs, || Expr::Var(b.slot));
                                out.push(Stmt::Set(b.slot, value));
                            }
                            // Assignment to an undeclared name = an implicit global
                            // write; passes through to the residual.
                            None => {
                                let value = combine_assign(&op_combine, rhs, || {
                                    Expr::Global(name.to_string())
                                });
                                out.push(Stmt::SetGlobal(name.to_string(), value));
                            }
                        }
                    }
                    ast::AssignTarget::Simple(ast::SimpleAssignTarget::Member(m)) => {
                        let obj = self.lower_expr(&m.obj, ctx)?;
                        match &m.prop {
                            ast::MemberProp::Ident(id) => {
                                let key = id.sym.to_string();
                                let value = combine_assign(&op_combine, rhs, || {
                                    Expr::Get(Box::new(obj.clone()), key.clone())
                                });
                                out.push(Stmt::SetProp(obj, key, value));
                            }
                            ast::MemberProp::Computed(c) => {
                                let idx = self.lower_expr(&c.expr, ctx)?;
                                let value = combine_assign(&op_combine, rhs, || {
                                    Expr::Index(Box::new(obj.clone()), Box::new(idx.clone()))
                                });
                                out.push(Stmt::SetIndex(obj, idx, value));
                            }
                            ast::MemberProp::PrivateName(_) => {
                                return Err("private fields are not supported".into())
                            }
                        }
                    }
                    _ => return Err("unsupported assignment target (destructuring?)".into()),
                }
            }
            ast::Expr::Call(c) => {
                // `arr.push(x)` is the one statement-form method call we support.
                if let Some((arr, args)) = self.as_push_call(c, ctx)? {
                    if args.len() != 1 {
                        return Err("`push` takes exactly one argument in this subset".into());
                    }
                    out.push(Stmt::Push(arr, args.into_iter().next().unwrap()));
                } else {
                    let call = self.lower_call(c, ctx)?;
                    out.push(Stmt::ExprStmt(call));
                }
            }
            // `delete obj.k` / `delete obj[i]` as a statement (the common form).
            ast::Expr::Unary(u) if matches!(u.op, ast::UnaryOp::Delete) => {
                let member = match &*u.arg {
                    ast::Expr::Member(m) => m,
                    _ => return Err("`delete` of a non-member expression is not supported".into()),
                };
                let obj = self.lower_expr(&member.obj, ctx)?;
                match &member.prop {
                    ast::MemberProp::Ident(id) => {
                        out.push(Stmt::DeleteProp(obj, id.sym.to_string()));
                    }
                    ast::MemberProp::Computed(c) => {
                        let idx = self.lower_expr(&c.expr, ctx)?;
                        out.push(Stmt::DeleteIndex(obj, idx));
                    }
                    ast::MemberProp::PrivateName(_) => {
                        return Err("private fields are not supported".into())
                    }
                }
            }
            ast::Expr::Paren(p) => self.lower_expr_stmt(&p.expr, ctx, out)?,
            other => {
                // any other expression statement: lower as a (side-effect-free) expr
                let lowered = self.lower_expr(other, ctx)?;
                out.push(Stmt::ExprStmt(lowered));
            }
        }
        Ok(())
    }

    // ---- expressions ----

    fn lower_expr(&mut self, e: &ast::Expr, ctx: &mut FnCtx) -> R<Expr> {
        match e {
            ast::Expr::Lit(lit) => lower_lit(lit),
            ast::Expr::Ident(id) => self.lower_ident(id.sym.as_str(), ctx),
            ast::Expr::Paren(p) => self.lower_expr(&p.expr, ctx),
            ast::Expr::Bin(b) => {
                // Operators we model and constant-fold:
                let modeled = match b.op {
                    ast::BinaryOp::Add => Some(Bop::Add),
                    ast::BinaryOp::Sub => Some(Bop::Sub),
                    ast::BinaryOp::Mul => Some(Bop::Mul),
                    ast::BinaryOp::Lt => Some(Bop::Lt),
                    ast::BinaryOp::LtEq => Some(Bop::Le),
                    ast::BinaryOp::Gt => Some(Bop::Gt),
                    ast::BinaryOp::GtEq => Some(Bop::Ge),
                    ast::BinaryOp::EqEqEq => Some(Bop::Eq),
                    ast::BinaryOp::NotEqEq => Some(Bop::Ne),
                    // Bitwise / shift: modeled so they fold on static integers
                    // (JS 32-bit semantics) and residualize when dynamic.
                    ast::BinaryOp::BitAnd => Some(Bop::BitAnd),
                    ast::BinaryOp::BitOr => Some(Bop::BitOr),
                    ast::BinaryOp::BitXor => Some(Bop::BitXor),
                    ast::BinaryOp::LShift => Some(Bop::Shl),
                    ast::BinaryOp::RShift => Some(Bop::Shr),
                    ast::BinaryOp::ZeroFillRShift => Some(Bop::UShr),
                    _ => None,
                };
                let l = self.lower_expr(&b.left, ctx)?;
                let r = self.lower_expr(&b.right, ctx)?;
                if let Some(op) = modeled {
                    return Ok(Expr::Bin(op, Box::new(l), Box::new(r)));
                }
                // Operators we don't model but that are pure over primitives:
                // pass them through verbatim (the operands are still specialized).
                let token = match b.op {
                    ast::BinaryOp::Div => "/",
                    ast::BinaryOp::Mod => "%",
                    ast::BinaryOp::Exp => "**",
                    ast::BinaryOp::LogicalAnd => "&&",
                    ast::BinaryOp::LogicalOr => "||",
                    ast::BinaryOp::NullishCoalescing => "??",
                    ast::BinaryOp::EqEq => "==",
                    ast::BinaryOp::NotEq => "!=",
                    // `in` / `instanceof` are pure (no mutation); an object
                    // operand escapes, then the test passes through verbatim.
                    ast::BinaryOp::In => "in",
                    ast::BinaryOp::InstanceOf => "instanceof",
                    other => return Err(format!("unsupported binary operator {:?}", other)),
                };
                Ok(Expr::Opaque(token.to_string(), vec![l, r]))
            }
            ast::Expr::Unary(u) => match u.op {
                ast::UnaryOp::Minus => {
                    let arg = self.lower_expr(&u.arg, ctx)?;
                    Ok(Expr::Bin(Bop::Sub, Box::new(Expr::Num(0)), Box::new(arg)))
                }
                // Unary `+x` is `ToNumber(x)`, exactly `x - 0` (unary `-` is
                // `0 - x`). Lowering it this way means the coercion is real (the
                // old identity lowering silently dropped it) and folds through the
                // same numeric-coercion path as subtraction.
                ast::UnaryOp::Plus => {
                    let arg = self.lower_expr(&u.arg, ctx)?;
                    Ok(Expr::Bin(Bop::Sub, Box::new(arg), Box::new(Expr::Num(0))))
                }
                // Pure unary operators pass through. (`void e` evaluates `e` and
                // yields undefined; emitting `(void e)` preserves both.)
                ast::UnaryOp::Bang
                | ast::UnaryOp::Tilde
                | ast::UnaryOp::TypeOf
                | ast::UnaryOp::Void => {
                    let token = match u.op {
                        ast::UnaryOp::Bang => "!",
                        ast::UnaryOp::Tilde => "~",
                        ast::UnaryOp::TypeOf => "typeof",
                        ast::UnaryOp::Void => "void",
                        _ => unreachable!(),
                    };
                    let arg = self.lower_expr(&u.arg, ctx)?;
                    Ok(Expr::Opaque(token.to_string(), vec![arg]))
                }
                ast::UnaryOp::Delete => {
                    Err("`delete` as a value is not supported (use it as a statement)".into())
                }
            },
            ast::Expr::Member(m) => {
                let obj = self.lower_expr(&m.obj, ctx)?;
                match &m.prop {
                    ast::MemberProp::Ident(id) => Ok(Expr::Get(Box::new(obj), id.sym.to_string())),
                    ast::MemberProp::Computed(c) => {
                        let idx = self.lower_expr(&c.expr, ctx)?;
                        Ok(Expr::Index(Box::new(obj), Box::new(idx)))
                    }
                    ast::MemberProp::PrivateName(_) => Err("private fields are not supported".into()),
                }
            }
            ast::Expr::Array(a) => {
                let mut elems = Vec::with_capacity(a.elems.len());
                for el in &a.elems {
                    match el {
                        Some(es) => {
                            if es.spread.is_some() {
                                return Err("array spread is not supported".into());
                            }
                            elems.push(self.lower_expr(&es.expr, ctx)?);
                        }
                        None => return Err("array holes are not supported".into()),
                    }
                }
                Ok(Expr::Array(elems))
            }
            ast::Expr::Object(o) => {
                let mut fields = Vec::with_capacity(o.props.len());
                for p in &o.props {
                    match p {
                        ast::PropOrSpread::Prop(prop) => match &**prop {
                            ast::Prop::KeyValue(kv) => {
                                let key = prop_name(&kv.key)?;
                                let value = self.lower_expr(&kv.value, ctx)?;
                                fields.push((key, value));
                            }
                            ast::Prop::Shorthand(id) => {
                                let name = id.sym.to_string();
                                let value = self.lower_ident(&name, ctx)?;
                                fields.push((name, value));
                            }
                            _ => return Err("only key:value and shorthand object properties are supported".into()),
                        },
                        ast::PropOrSpread::Spread(_) => {
                            return Err("object spread is not supported".into())
                        }
                    }
                }
                Ok(Expr::Object(fields))
            }
            ast::Expr::Call(c) => self.lower_call(c, ctx),
            ast::Expr::Arrow(a) => {
                let body = match &*a.body {
                    ast::BlockStmtOrExpr::BlockStmt(b) => ArrowBody::Block(&b.stmts),
                    ast::BlockStmtOrExpr::Expr(e) => ArrowBody::Expr(e),
                };
                let (fid, captures) = self.lift_arrow(&a.params, body, ctx)?;
                let caps = self.capture_exprs(&captures, ctx)?;
                Ok(Expr::Closure(fid, caps))
            }
            ast::Expr::Fn(f) => {
                let params: Vec<ast::Pat> =
                    f.function.params.iter().map(|p| p.pat.clone()).collect();
                let body_stmts = match &f.function.body {
                    Some(b) => b.stmts.clone(),
                    None => return Err("function expression without a body".into()),
                };
                // A named function expression's name is in scope only inside its
                // own body (for self-reference). If the body doesn't actually use
                // it as a free variable, the name is cosmetic and we lift it as
                // anonymous; if it does, that is self-recursion (out of scope).
                if let Some(id) = &f.ident {
                    let name = id.sym.to_string();
                    if free_vars(&params, &body_stmts).contains(&name) {
                        return Err(format!(
                            "self-referential named function expression `{name}` is not \
                             supported (recursion); use a top-level declaration"
                        ));
                    }
                }
                let (fid, captures) =
                    self.lift_arrow(&params, ArrowBody::BlockOwned(body_stmts), ctx)?;
                let caps = self.capture_exprs(&captures, ctx)?;
                Ok(Expr::Closure(fid, caps))
            }
            ast::Expr::Cond(c) => {
                // `test ? cons : alt` passes through; operands are specialized.
                let test = self.lower_expr(&c.test, ctx)?;
                let cons = self.lower_expr(&c.cons, ctx)?;
                let alt = self.lower_expr(&c.alt, ctx)?;
                Ok(Expr::Opaque("?:".to_string(), vec![test, cons, alt]))
            }
            ast::Expr::Tpl(_) => Err("template literals are not supported".into()),
            ast::Expr::New(n) => {
                let callee = self.lower_expr(&n.callee, ctx)?;
                let mut args = Vec::new();
                if let Some(arg_list) = &n.args {
                    for a in arg_list {
                        if a.spread.is_some() {
                            return Err("`new` with spread arguments is not supported".into());
                        }
                        args.push(self.lower_expr(&a.expr, ctx)?);
                    }
                }
                Ok(Expr::New(Box::new(callee), args))
            }
            ast::Expr::This(_) => Ok(Expr::This),
            ast::Expr::Update(u) => {
                // `place++` / `++place` used for its value.
                let op = match u.op {
                    ast::UpdateOp::PlusPlus => Bop::Add,
                    ast::UpdateOp::MinusMinus => Bop::Sub,
                };
                let place = match &*u.arg {
                    ast::Expr::Ident(id) => {
                        let name = id.sym.as_str();
                        let b = ctx
                            .resolve_binding(name)
                            .ok_or_else(|| format!("update of unbound variable `{name}`"))?;
                        if b.boxed {
                            Expr::Get(Box::new(Expr::Var(b.slot)), "value".to_string())
                        } else {
                            Expr::Var(b.slot)
                        }
                    }
                    ast::Expr::Member(m) => {
                        let obj = self.lower_expr(&m.obj, ctx)?;
                        match &m.prop {
                            ast::MemberProp::Ident(id) => {
                                Expr::Get(Box::new(obj), id.sym.to_string())
                            }
                            ast::MemberProp::Computed(c) => {
                                let idx = self.lower_expr(&c.expr, ctx)?;
                                Expr::Index(Box::new(obj), Box::new(idx))
                            }
                            ast::MemberProp::PrivateName(_) => {
                                return Err("private fields are not supported".into())
                            }
                        }
                    }
                    _ => return Err("`++`/`--` on an unsupported target".into()),
                };
                Ok(Expr::Update { place: Box::new(place), op, prefix: u.prefix })
            }
            ast::Expr::Assign(_) => Err("assignment as an expression is not supported (use a statement)".into()),
            other => Err(format!("unsupported expression: {:?}", other)),
        }
    }

    fn lower_ident(&self, name: &str, ctx: &mut FnCtx) -> R<Expr> {
        if let Some(b) = ctx.resolve_binding(name) {
            // A boxed variable lives in a `{value}` cell; read through it.
            Ok(if b.boxed {
                Expr::Get(Box::new(Expr::Var(b.slot)), "value".to_string())
            } else {
                Expr::Var(b.slot)
            })
        } else if let Some(&fid) = self.name_to_fid.get(name) {
            Ok(Expr::Func(fid))
        } else if name == "undefined" {
            Ok(Expr::Undefined)
        } else if name == "arguments" {
            // `arguments` reads from a per-function slot the engine fills with
            // the actual call args (see `FuncDef::arguments_slot`). Reserved
            // lazily so only functions that use it pay for it.
            Ok(Expr::Var(ctx.arguments_slot_or_reserve()))
        } else if ctx.outer_names.contains(name) {
            Err(format!(
                "identifier `{name}` is captured from an enclosing scope; closures \
                 capturing outer variables are not supported in this subset"
            ))
        } else {
            // Not a local, a known function, or a capture: treat it as a runtime
            // global (e.g. `Math`, `parseInt`) and let it pass through.
            Ok(Expr::Global(name.to_string()))
        }
    }

    fn lower_call(&mut self, c: &ast::CallExpr, ctx: &mut FnCtx) -> R<Expr> {
        // The callee is lowered like any expression: a known function name
        // becomes `Func`/`Var` (a modeled, inlinable call); a member access
        // (`recv.method`) becomes a `Get`, and an unknown name a `Global`, both
        // of which evaluate to a dynamic callee and pass the call through.
        // (`arr.push(x)` in statement position is special-cased earlier.)
        let callee = match &c.callee {
            ast::Callee::Expr(e) => self.lower_expr(e, ctx)?,
            _ => return Err("`super(...)` / `import(...)` are not supported".into()),
        };
        let mut args = Vec::with_capacity(c.args.len());
        for a in &c.args {
            if a.spread.is_some() {
                return Err("call spread is not supported".into());
            }
            args.push(self.lower_expr(&a.expr, ctx)?);
        }
        Ok(Expr::Call(Box::new(callee), args))
    }

    /// Recognize `arr.push(x)`; returns `(arr_expr, args)` if it matches.
    fn as_push_call(&mut self, c: &ast::CallExpr, ctx: &mut FnCtx) -> R<Option<(Expr, Vec<Expr>)>> {
        if let ast::Callee::Expr(e) = &c.callee {
            if let ast::Expr::Member(m) = &**e {
                if let ast::MemberProp::Ident(id) = &m.prop {
                    if id.sym.as_str() == "push" {
                        let arr = self.lower_expr(&m.obj, ctx)?;
                        let mut args = Vec::new();
                        for a in &c.args {
                            if a.spread.is_some() {
                                return Err("call spread is not supported".into());
                            }
                            args.push(self.lower_expr(&a.expr, ctx)?);
                        }
                        return Ok(Some((arr, args)));
                    }
                }
            }
        }
        Ok(None)
    }
}

enum ArrowBody<'a> {
    Block(&'a [ast::Stmt]),
    BlockOwned(Vec<ast::Stmt>),
    Expr(&'a ast::Expr),
}

/// Free variables of a function: identifiers it reads that are bound neither by
/// its parameters nor by any declaration inside it. (Used to compute closure
/// captures. Over-approximating `bound` only over-captures, which is harmless;
/// the result is later intersected with the names actually visible outside.)
/// How a compound assignment combines the current value with the rhs.
enum Combine {
    /// A modeled, constant-foldable binary op (`+`, `-`, `*`).
    Bin(Bop),
    /// An unmodeled-but-pure op passed through verbatim (`/`, `%`, bitwise, ...).
    Opaque(&'static str),
}

/// Build the value for `target <op>= rhs`. `cur` lazily constructs the read of
/// the target (only needed for a compound op, not a plain `=`).
fn combine_assign(op: &Option<Combine>, rhs: Expr, cur: impl FnOnce() -> Expr) -> Expr {
    match op {
        None => rhs,
        Some(Combine::Bin(op)) => Expr::Bin(*op, Box::new(cur()), Box::new(rhs)),
        Some(Combine::Opaque(tok)) => Expr::Opaque(tok.to_string(), vec![cur(), rhs]),
    }
}

fn free_vars(params: &[ast::Pat], body: &[ast::Stmt]) -> Vec<String> {
    let mut used = HashSet::new();
    let mut bound = HashSet::new();
    for p in params {
        bound_in_pat(p, &mut bound);
    }
    for s in body {
        used_in_stmt(s, &mut used);
        bound_in_stmt(s, &mut bound);
    }
    let mut free: Vec<String> = used.into_iter().filter(|n| !bound.contains(n)).collect();
    free.sort();
    free
}

fn bound_in_pat(p: &ast::Pat, out: &mut HashSet<String>) {
    if let ast::Pat::Ident(bi) = p {
        out.insert(bi.id.sym.to_string());
    }
}

fn bound_in_stmt(s: &ast::Stmt, out: &mut HashSet<String>) {
    match s {
        ast::Stmt::Decl(ast::Decl::Var(v)) => {
            for d in &v.decls {
                bound_in_pat(&d.name, out);
            }
        }
        ast::Stmt::Decl(ast::Decl::Fn(f)) => {
            out.insert(f.ident.sym.to_string());
            if let Some(b) = &f.function.body {
                for p in &f.function.params {
                    bound_in_pat(&p.pat, out);
                }
                b.stmts.iter().for_each(|s| bound_in_stmt(s, out));
            }
        }
        ast::Stmt::Block(b) => b.stmts.iter().for_each(|s| bound_in_stmt(s, out)),
        ast::Stmt::If(i) => {
            bound_in_stmt(&i.cons, out);
            if let Some(a) = &i.alt {
                bound_in_stmt(a, out);
            }
        }
        ast::Stmt::For(f) => {
            if let Some(ast::VarDeclOrExpr::VarDecl(v)) = &f.init {
                for d in &v.decls {
                    bound_in_pat(&d.name, out);
                }
            }
            bound_in_stmt(&f.body, out);
        }
        ast::Stmt::ForIn(f) => bound_in_stmt(&f.body, out),
        ast::Stmt::ForOf(f) => bound_in_stmt(&f.body, out),
        ast::Stmt::While(w) => bound_in_stmt(&w.body, out),
        ast::Stmt::DoWhile(d) => bound_in_stmt(&d.body, out),
        ast::Stmt::Switch(sw) => {
            for c in &sw.cases {
                c.cons.iter().for_each(|s| bound_in_stmt(s, out));
            }
        }
        ast::Stmt::Try(t) => {
            t.block.stmts.iter().for_each(|s| bound_in_stmt(s, out));
            if let Some(h) = &t.handler {
                if let Some(p) = &h.param {
                    bound_in_pat(p, out);
                }
                h.body.stmts.iter().for_each(|s| bound_in_stmt(s, out));
            }
            if let Some(f) = &t.finalizer {
                f.stmts.iter().for_each(|s| bound_in_stmt(s, out));
            }
        }
        ast::Stmt::Labeled(l) => bound_in_stmt(&l.body, out),
        ast::Stmt::Expr(e) => bound_in_expr(&e.expr, out),
        ast::Stmt::Return(r) => {
            if let Some(e) = &r.arg {
                bound_in_expr(e, out);
            }
        }
        _ => {}
    }
}

/// Bindings introduced by function/arrow *expressions* (their params + inner
/// declarations) — these are bound from the perspective of the enclosing scope.
fn bound_in_expr(e: &ast::Expr, out: &mut HashSet<String>) {
    match e {
        ast::Expr::Fn(f) => {
            if let Some(b) = &f.function.body {
                for p in &f.function.params {
                    bound_in_pat(&p.pat, out);
                }
                b.stmts.iter().for_each(|s| bound_in_stmt(s, out));
            }
        }
        ast::Expr::Arrow(a) => {
            for p in &a.params {
                bound_in_pat(p, out);
            }
            match &*a.body {
                ast::BlockStmtOrExpr::BlockStmt(b) => {
                    b.stmts.iter().for_each(|s| bound_in_stmt(s, out))
                }
                ast::BlockStmtOrExpr::Expr(e) => bound_in_expr(e, out),
            }
        }
        ast::Expr::Paren(p) => bound_in_expr(&p.expr, out),
        ast::Expr::Call(c) => {
            if let ast::Callee::Expr(e) = &c.callee {
                bound_in_expr(e, out);
            }
            for a in &c.args {
                bound_in_expr(&a.expr, out);
            }
        }
        ast::Expr::Bin(b) => {
            bound_in_expr(&b.left, out);
            bound_in_expr(&b.right, out);
        }
        ast::Expr::Assign(a) => bound_in_expr(&a.right, out),
        _ => {}
    }
}

fn used_in_stmt(s: &ast::Stmt, out: &mut HashSet<String>) {
    match s {
        ast::Stmt::Expr(e) => used_in_expr(&e.expr, out),
        ast::Stmt::Decl(ast::Decl::Var(v)) => {
            for d in &v.decls {
                if let Some(e) = &d.init {
                    used_in_expr(e, out);
                }
            }
        }
        ast::Stmt::Decl(ast::Decl::Fn(f)) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        ast::Stmt::Return(r) => {
            if let Some(e) = &r.arg {
                used_in_expr(e, out);
            }
        }
        ast::Stmt::Block(b) => b.stmts.iter().for_each(|s| used_in_stmt(s, out)),
        ast::Stmt::If(i) => {
            used_in_expr(&i.test, out);
            used_in_stmt(&i.cons, out);
            if let Some(a) = &i.alt {
                used_in_stmt(a, out);
            }
        }
        ast::Stmt::For(f) => {
            match &f.init {
                Some(ast::VarDeclOrExpr::VarDecl(v)) => {
                    for d in &v.decls {
                        if let Some(e) = &d.init {
                            used_in_expr(e, out);
                        }
                    }
                }
                Some(ast::VarDeclOrExpr::Expr(e)) => used_in_expr(e, out),
                None => {}
            }
            if let Some(t) = &f.test {
                used_in_expr(t, out);
            }
            if let Some(u) = &f.update {
                used_in_expr(u, out);
            }
            used_in_stmt(&f.body, out);
        }
        ast::Stmt::ForIn(f) => {
            used_in_expr(&f.right, out);
            used_in_stmt(&f.body, out);
        }
        ast::Stmt::ForOf(f) => {
            used_in_expr(&f.right, out);
            used_in_stmt(&f.body, out);
        }
        ast::Stmt::While(w) => {
            used_in_expr(&w.test, out);
            used_in_stmt(&w.body, out);
        }
        ast::Stmt::DoWhile(d) => {
            used_in_expr(&d.test, out);
            used_in_stmt(&d.body, out);
        }
        ast::Stmt::Switch(sw) => {
            used_in_expr(&sw.discriminant, out);
            for c in &sw.cases {
                if let Some(t) = &c.test {
                    used_in_expr(t, out);
                }
                c.cons.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        ast::Stmt::Throw(t) => used_in_expr(&t.arg, out),
        ast::Stmt::Try(t) => {
            t.block.stmts.iter().for_each(|s| used_in_stmt(s, out));
            if let Some(h) = &t.handler {
                h.body.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
            if let Some(f) = &t.finalizer {
                f.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        ast::Stmt::Labeled(l) => used_in_stmt(&l.body, out),
        _ => {}
    }
}

fn used_in_expr(e: &ast::Expr, out: &mut HashSet<String>) {
    match e {
        ast::Expr::Ident(i) => {
            out.insert(i.sym.to_string());
        }
        ast::Expr::Bin(b) => {
            used_in_expr(&b.left, out);
            used_in_expr(&b.right, out);
        }
        ast::Expr::Unary(u) => used_in_expr(&u.arg, out),
        ast::Expr::Update(u) => used_in_expr(&u.arg, out),
        ast::Expr::Cond(c) => {
            used_in_expr(&c.test, out);
            used_in_expr(&c.cons, out);
            used_in_expr(&c.alt, out);
        }
        ast::Expr::Member(m) => {
            used_in_expr(&m.obj, out);
            if let ast::MemberProp::Computed(c) = &m.prop {
                used_in_expr(&c.expr, out);
            }
        }
        ast::Expr::Call(c) => {
            if let ast::Callee::Expr(e) = &c.callee {
                used_in_expr(e, out);
            }
            for a in &c.args {
                used_in_expr(&a.expr, out);
            }
        }
        ast::Expr::New(n) => {
            used_in_expr(&n.callee, out);
            if let Some(args) = &n.args {
                for a in args {
                    used_in_expr(&a.expr, out);
                }
            }
        }
        ast::Expr::Array(a) => {
            for el in a.elems.iter().flatten() {
                used_in_expr(&el.expr, out);
            }
        }
        ast::Expr::Object(o) => {
            for p in &o.props {
                if let ast::PropOrSpread::Prop(prop) = p {
                    match &**prop {
                        ast::Prop::KeyValue(kv) => used_in_expr(&kv.value, out),
                        ast::Prop::Shorthand(id) => {
                            out.insert(id.sym.to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
        ast::Expr::Paren(p) => used_in_expr(&p.expr, out),
        ast::Expr::Assign(a) => {
            // a simple-ident assignment target is also a use of that binding
            if let ast::AssignTarget::Simple(ast::SimpleAssignTarget::Ident(bi)) = &a.left {
                out.insert(bi.id.sym.to_string());
            }
            if let ast::AssignTarget::Simple(ast::SimpleAssignTarget::Member(m)) = &a.left {
                used_in_expr(&m.obj, out);
                if let ast::MemberProp::Computed(c) = &m.prop {
                    used_in_expr(&c.expr, out);
                }
            }
            used_in_expr(&a.right, out);
        }
        // A nested function/arrow's free uses are uses in the enclosing scope too.
        ast::Expr::Arrow(a) => match &*a.body {
            ast::BlockStmtOrExpr::BlockStmt(b) => {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out))
            }
            ast::BlockStmtOrExpr::Expr(e) => used_in_expr(e, out),
        },
        ast::Expr::Fn(f) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        _ => {}
    }
}

/// Variables of a function that must be BOXED for capture-by-reference: a `var`
/// that is both used inside a nested function (captured) AND assigned somewhere
/// (mutated). Boxing makes them shared `{value: ...}` cells so a closure's
/// mutation is visible to everyone. (Only `var`s are boxed; a captured+mutated
/// `let`/`const` is rejected at lowering.)
fn boxed_locals(params: &[&str], body: &[ast::Stmt]) -> HashSet<String> {
    let mut vars = Vec::new();
    for s in body {
        collect_vars(s, &mut vars);
    }
    // Parameters are capture candidates too: a captured + reassigned parameter
    // (reassigned by the outer body or by the capturing closure) must be a shared
    // cell, exactly like a captured + mutated `var`. Without this it captures
    // by-value and the reassignment is invisible across the closure boundary.
    let mut vars: HashSet<String> = vars.into_iter().collect();
    vars.extend(params.iter().map(|p| p.to_string()));
    let mut used_nested = HashSet::new();
    for s in body {
        nested_used_stmt(s, &mut used_nested);
    }
    let mut assigned = HashSet::new();
    for s in body {
        assigned_stmt(s, &mut assigned);
    }
    let mut boxed: HashSet<String> = vars
        .into_iter()
        .filter(|n| used_nested.contains(n) && assigned.contains(n))
        .collect();
    // A nested `function` declaration whose name is referenced by another nested
    // function (e.g. mutual recursion) must also be a shared cell: its value is
    // assigned after the capturing closures are created, so by-value capture
    // would snapshot `undefined`. Boxing makes the later assignment visible.
    for name in fn_decl_names(body) {
        if used_nested.contains(&name) {
            boxed.insert(name);
        }
    }
    boxed
}

/// Names of `function` declarations that are direct children of `stmts` (the top
/// level of a function body). Block-nested function declarations are not hoisted
/// here and are rejected at lowering.
fn fn_decl_names(stmts: &[ast::Stmt]) -> Vec<String> {
    stmts
        .iter()
        .filter_map(|s| match s {
            ast::Stmt::Decl(ast::Decl::Fn(f)) => Some(f.ident.sym.to_string()),
            _ => None,
        })
        .collect()
}

/// Identifiers used *inside nested functions* of the current function (so we can
/// tell which of our locals are captured). At each nested function we collect
/// all of its used identifiers and stop descending (it handles its own nesting).
fn nested_used_stmt(s: &ast::Stmt, out: &mut HashSet<String>) {
    match s {
        ast::Stmt::Decl(ast::Decl::Fn(f)) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        ast::Stmt::Decl(ast::Decl::Var(v)) => {
            for d in &v.decls {
                if let Some(e) = &d.init {
                    nested_used_expr(e, out);
                }
            }
        }
        ast::Stmt::Expr(e) => nested_used_expr(&e.expr, out),
        ast::Stmt::Return(r) => {
            if let Some(e) = &r.arg {
                nested_used_expr(e, out);
            }
        }
        ast::Stmt::Throw(t) => nested_used_expr(&t.arg, out),
        ast::Stmt::Block(b) => b.stmts.iter().for_each(|s| nested_used_stmt(s, out)),
        ast::Stmt::If(i) => {
            nested_used_expr(&i.test, out);
            nested_used_stmt(&i.cons, out);
            if let Some(a) = &i.alt {
                nested_used_stmt(a, out);
            }
        }
        ast::Stmt::For(f) => {
            if let Some(ast::VarDeclOrExpr::VarDecl(v)) = &f.init {
                for d in &v.decls {
                    if let Some(e) = &d.init {
                        nested_used_expr(e, out);
                    }
                }
            } else if let Some(ast::VarDeclOrExpr::Expr(e)) = &f.init {
                nested_used_expr(e, out);
            }
            if let Some(t) = &f.test {
                nested_used_expr(t, out);
            }
            if let Some(u) = &f.update {
                nested_used_expr(u, out);
            }
            nested_used_stmt(&f.body, out);
        }
        ast::Stmt::While(w) => {
            nested_used_expr(&w.test, out);
            nested_used_stmt(&w.body, out);
        }
        ast::Stmt::DoWhile(d) => {
            nested_used_expr(&d.test, out);
            nested_used_stmt(&d.body, out);
        }
        ast::Stmt::Switch(sw) => {
            nested_used_expr(&sw.discriminant, out);
            for c in &sw.cases {
                if let Some(t) = &c.test {
                    nested_used_expr(t, out);
                }
                c.cons.iter().for_each(|s| nested_used_stmt(s, out));
            }
        }
        ast::Stmt::Try(t) => {
            t.block.stmts.iter().for_each(|s| nested_used_stmt(s, out));
            if let Some(h) = &t.handler {
                h.body.stmts.iter().for_each(|s| nested_used_stmt(s, out));
            }
            if let Some(f) = &t.finalizer {
                f.stmts.iter().for_each(|s| nested_used_stmt(s, out));
            }
        }
        ast::Stmt::Labeled(l) => nested_used_stmt(&l.body, out),
        _ => {}
    }
}

fn nested_used_expr(e: &ast::Expr, out: &mut HashSet<String>) {
    match e {
        // a nested function: collect everything it uses, then stop
        ast::Expr::Fn(f) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out));
            }
        }
        ast::Expr::Arrow(a) => match &*a.body {
            ast::BlockStmtOrExpr::BlockStmt(b) => {
                b.stmts.iter().for_each(|s| used_in_stmt(s, out))
            }
            ast::BlockStmtOrExpr::Expr(e) => used_in_expr(e, out),
        },
        ast::Expr::Bin(b) => {
            nested_used_expr(&b.left, out);
            nested_used_expr(&b.right, out);
        }
        ast::Expr::Unary(u) => nested_used_expr(&u.arg, out),
        ast::Expr::Update(u) => nested_used_expr(&u.arg, out),
        ast::Expr::Cond(c) => {
            nested_used_expr(&c.test, out);
            nested_used_expr(&c.cons, out);
            nested_used_expr(&c.alt, out);
        }
        ast::Expr::Member(m) => {
            nested_used_expr(&m.obj, out);
            if let ast::MemberProp::Computed(c) = &m.prop {
                nested_used_expr(&c.expr, out);
            }
        }
        ast::Expr::Call(c) => {
            if let ast::Callee::Expr(e) = &c.callee {
                nested_used_expr(e, out);
            }
            for a in &c.args {
                nested_used_expr(&a.expr, out);
            }
        }
        ast::Expr::New(n) => {
            nested_used_expr(&n.callee, out);
            if let Some(args) = &n.args {
                for a in args {
                    nested_used_expr(&a.expr, out);
                }
            }
        }
        ast::Expr::Array(a) => {
            for el in a.elems.iter().flatten() {
                nested_used_expr(&el.expr, out);
            }
        }
        ast::Expr::Object(o) => {
            for p in &o.props {
                if let ast::PropOrSpread::Prop(prop) = p {
                    if let ast::Prop::KeyValue(kv) = &**prop {
                        nested_used_expr(&kv.value, out);
                    }
                }
            }
        }
        ast::Expr::Paren(p) => nested_used_expr(&p.expr, out),
        ast::Expr::Assign(a) => nested_used_expr(&a.right, out),
        _ => {}
    }
}

/// Names that are assignment / update targets anywhere (descending into nested
/// functions, since a closure may mutate an enclosing variable).
fn assigned_stmt(s: &ast::Stmt, out: &mut HashSet<String>) {
    match s {
        ast::Stmt::Expr(e) => assigned_expr(&e.expr, out),
        ast::Stmt::Decl(ast::Decl::Var(v)) => {
            for d in &v.decls {
                if let Some(e) = &d.init {
                    assigned_expr(e, out);
                }
            }
        }
        ast::Stmt::Decl(ast::Decl::Fn(f)) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| assigned_stmt(s, out));
            }
        }
        ast::Stmt::Return(r) => {
            if let Some(e) = &r.arg {
                assigned_expr(e, out);
            }
        }
        ast::Stmt::Throw(t) => assigned_expr(&t.arg, out),
        ast::Stmt::Block(b) => b.stmts.iter().for_each(|s| assigned_stmt(s, out)),
        ast::Stmt::If(i) => {
            assigned_expr(&i.test, out);
            assigned_stmt(&i.cons, out);
            if let Some(a) = &i.alt {
                assigned_stmt(a, out);
            }
        }
        ast::Stmt::For(f) => {
            match &f.init {
                Some(ast::VarDeclOrExpr::VarDecl(v)) => {
                    for d in &v.decls {
                        if let Some(e) = &d.init {
                            assigned_expr(e, out);
                        }
                    }
                }
                Some(ast::VarDeclOrExpr::Expr(e)) => assigned_expr(e, out),
                None => {}
            }
            if let Some(u) = &f.update {
                assigned_expr(u, out);
            }
            assigned_stmt(&f.body, out);
        }
        ast::Stmt::While(w) => {
            assigned_expr(&w.test, out);
            assigned_stmt(&w.body, out);
        }
        ast::Stmt::DoWhile(d) => assigned_stmt(&d.body, out),
        ast::Stmt::Switch(sw) => {
            for c in &sw.cases {
                c.cons.iter().for_each(|s| assigned_stmt(s, out));
            }
        }
        ast::Stmt::Try(t) => {
            t.block.stmts.iter().for_each(|s| assigned_stmt(s, out));
            if let Some(h) = &t.handler {
                h.body.stmts.iter().for_each(|s| assigned_stmt(s, out));
            }
            if let Some(f) = &t.finalizer {
                f.stmts.iter().for_each(|s| assigned_stmt(s, out));
            }
        }
        ast::Stmt::Labeled(l) => assigned_stmt(&l.body, out),
        _ => {}
    }
}

fn assigned_expr(e: &ast::Expr, out: &mut HashSet<String>) {
    match e {
        ast::Expr::Assign(a) => {
            if let ast::AssignTarget::Simple(ast::SimpleAssignTarget::Ident(bi)) = &a.left {
                out.insert(bi.id.sym.to_string());
            }
            assigned_expr(&a.right, out);
        }
        ast::Expr::Update(u) => {
            if let ast::Expr::Ident(id) = &*u.arg {
                out.insert(id.sym.to_string());
            }
        }
        ast::Expr::Bin(b) => {
            assigned_expr(&b.left, out);
            assigned_expr(&b.right, out);
        }
        ast::Expr::Unary(u) => assigned_expr(&u.arg, out),
        ast::Expr::Cond(c) => {
            assigned_expr(&c.test, out);
            assigned_expr(&c.cons, out);
            assigned_expr(&c.alt, out);
        }
        ast::Expr::Member(m) => {
            assigned_expr(&m.obj, out);
            // A computed property can itself contain an assignment/update, e.g.
            // `arr[i++]` mutates `i`; missing this would leave `i` un-boxed and
            // captured by value, dropping the increment.
            if let ast::MemberProp::Computed(c) = &m.prop {
                assigned_expr(&c.expr, out);
            }
        }
        ast::Expr::Call(c) => {
            if let ast::Callee::Expr(e) = &c.callee {
                assigned_expr(e, out);
            }
            for a in &c.args {
                assigned_expr(&a.expr, out);
            }
        }
        ast::Expr::New(n) => {
            if let Some(args) = &n.args {
                for a in args {
                    assigned_expr(&a.expr, out);
                }
            }
        }
        ast::Expr::Array(a) => {
            for el in a.elems.iter().flatten() {
                assigned_expr(&el.expr, out);
            }
        }
        ast::Expr::Object(o) => {
            for p in &o.props {
                if let ast::PropOrSpread::Prop(prop) = p {
                    if let ast::Prop::KeyValue(kv) = &**prop {
                        assigned_expr(&kv.value, out);
                    }
                }
            }
        }
        ast::Expr::Paren(p) => assigned_expr(&p.expr, out),
        // descend into nested functions: they may mutate our variables
        ast::Expr::Fn(f) => {
            if let Some(b) = &f.function.body {
                b.stmts.iter().for_each(|s| assigned_stmt(s, out));
            }
        }
        ast::Expr::Arrow(a) => match &*a.body {
            ast::BlockStmtOrExpr::BlockStmt(b) => {
                b.stmts.iter().for_each(|s| assigned_stmt(s, out))
            }
            ast::BlockStmtOrExpr::Expr(e) => assigned_expr(e, out),
        },
        _ => {}
    }
}

/// Pre-declare every `var` in a function body so it is visible throughout
/// (JS `var` hoisting: usable, as `undefined`, before its declaration line).
/// Recurses into nested statement bodies but NOT into nested functions, which
/// have their own `var` scope.
fn hoist_vars(stmts: &[ast::Stmt], ctx: &mut FnCtx) {
    let mut names = Vec::new();
    for s in stmts {
        collect_vars(s, &mut names);
    }
    for name in names {
        ctx.declare_var(&name);
    }
}

fn collect_decl_idents(v: &ast::VarDecl, out: &mut Vec<String>) {
    if !matches!(v.kind, ast::VarDeclKind::Var) {
        return; // let/const are block-scoped, not hoisted
    }
    for d in &v.decls {
        if let ast::Pat::Ident(bi) = &d.name {
            out.push(bi.id.sym.to_string());
        }
    }
}

fn collect_vars(s: &ast::Stmt, out: &mut Vec<String>) {
    match s {
        ast::Stmt::Decl(ast::Decl::Var(v)) => collect_decl_idents(v, out),
        ast::Stmt::Block(b) => b.stmts.iter().for_each(|s| collect_vars(s, out)),
        ast::Stmt::If(i) => {
            collect_vars(&i.cons, out);
            if let Some(a) = &i.alt {
                collect_vars(a, out);
            }
        }
        ast::Stmt::For(f) => {
            if let Some(ast::VarDeclOrExpr::VarDecl(v)) = &f.init {
                collect_decl_idents(v, out);
            }
            collect_vars(&f.body, out);
        }
        ast::Stmt::ForIn(f) => collect_vars(&f.body, out),
        ast::Stmt::ForOf(f) => collect_vars(&f.body, out),
        ast::Stmt::While(w) => collect_vars(&w.body, out),
        ast::Stmt::DoWhile(d) => collect_vars(&d.body, out),
        ast::Stmt::Switch(sw) => {
            for c in &sw.cases {
                c.cons.iter().for_each(|s| collect_vars(s, out));
            }
        }
        ast::Stmt::Try(t) => {
            t.block.stmts.iter().for_each(|s| collect_vars(s, out));
            if let Some(h) = &t.handler {
                h.body.stmts.iter().for_each(|s| collect_vars(s, out));
            }
            if let Some(f) = &t.finalizer {
                f.stmts.iter().for_each(|s| collect_vars(s, out));
            }
        }
        ast::Stmt::Labeled(l) => collect_vars(&l.body, out),
        // expressions and nested functions don't contribute function-scoped vars
        _ => {}
    }
}

fn lower_lit(lit: &ast::Lit) -> R<Expr> {
    match lit {
        ast::Lit::Num(n) => {
            if !n.value.is_finite() || n.value.fract() != 0.0 {
                return Err(format!(
                    "only integer number literals are supported (got {})",
                    n.value
                ));
            }
            Ok(Expr::Num(n.value as i64))
        }
        ast::Lit::Str(s) => Ok(Expr::Str(s.value.to_string_lossy().into_owned())),
        ast::Lit::Bool(b) => Ok(Expr::Bool(b.value)),
        ast::Lit::Null(_) => Ok(Expr::Null),
        ast::Lit::BigInt(_) => Err("BigInt literals are not supported".into()),
        ast::Lit::Regex(_) => Err("regex literals are not supported".into()),
        ast::Lit::JSXText(_) => Err("JSX is not supported".into()),
    }
}

fn prop_name(p: &ast::PropName) -> R<String> {
    match p {
        ast::PropName::Ident(id) => Ok(id.sym.to_string()),
        ast::PropName::Str(s) => Ok(s.value.to_string_lossy().into_owned()),
        ast::PropName::Num(n) => Ok((n.value as i64).to_string()),
        _ => Err("computed / BigInt object keys are not supported".into()),
    }
}

fn pat_ident(p: &ast::Pat) -> R<&str> {
    match p {
        ast::Pat::Ident(bi) => Ok(bi.id.sym.as_str()),
        _ => Err("only simple identifier bindings are supported (no destructuring)".into()),
    }
}
