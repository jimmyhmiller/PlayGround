//! swc AST -> IR, with **no intermediate tree**.
//!
//! `ast2hir` is generic over [`jsir_ast::AstNode`]; here we implement that exact
//! trait over swc's own AST, so a parsed swc `Program` lowers straight to JSIR
//! IR. The orphan rule forbids implementing a foreign trait on swc's foreign
//! types directly, so a single thin wrapper [`W`] borrows any swc node (or a
//! sub-record we need to synthesize, like a literal's `extra`) and presents it
//! with Babel node-type tags and field keys. Child wrappers are materialized on
//! demand in a bump arena (no owned parallel AST is built); the arena and the
//! parsed program outlive a single `source_to_ir` call.

use bumpalo::Bump;
use jsir_ast::{AstNode, Field};
use std::collections::HashMap;
use swc_common::comments::{Comment, CommentKind, SingleThreadedComments};
use swc_common::{sync::Lrc, BytePos, FileName, SourceMap, Span, Spanned};
use swc_ecma_ast as ast;
use swc_ecma_ast::EsVersion;
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};
use swc_ecma_visit::{Visit, VisitWith};

/// Source-position context shared by every wrapper: the swc source map (turns
/// byte positions into line/column), the parsed file's start position (turns
/// byte positions into 0-based source offsets), and the Babel-compatible
/// `scopeUid` for every identifier span.
pub struct Cx {
    file_start: BytePos,
    /// The source text, for byte -> UTF-16 offset conversion (Babel offsets are
    /// UTF-16 code units, not bytes).
    src: String,
    /// UTF-16 offset at which each line starts (line 1 = index 0). Lines break on
    /// `\n`, `\r`, `\r\n`, `U+2028`, `U+2029` — Babel's terminator set, which swc
    /// does not fully share, so we compute line/column ourselves.
    line_starts: Vec<u32>,
    scopes: HashMap<(u32, u32), u32>,
    /// `var` binding ident span -> hoisted defScopeUid (function/program scope).
    var_defs: HashMap<(u32, u32), u32>,
    /// Scope tree (scope -> parent) for resolving `referencedSymbol`.
    parent: HashMap<u32, u32>,
    /// Names declared in each scope.
    decls: HashMap<u32, std::collections::HashSet<String>>,
    /// All comments in source order (Babel's `File.comments`).
    comments: Vec<Comment>,
}

impl Cx {
    /// `(line, column)` for a byte position: 1-based line, 0-based column, in
    /// UTF-16 units from the line start, matching Babel's `loc`.
    fn line_col(&self, bp: BytePos) -> (i64, i64) {
        let u = self.offset(bp) as u32;
        // Largest line index whose start is <= u (line_starts is sorted).
        let line = self.line_starts.partition_point(|&s| s <= u);
        let line = line.max(1);
        let col = u - self.line_starts[line - 1];
        (line as i64, col as i64)
    }
    /// 0-based source offset for a byte position, in **UTF-16 code units** to
    /// match Babel's `start`/`end` (which index the source as a JS string).
    /// swc reports byte positions, which diverge after any non-ASCII character.
    fn offset(&self, bp: BytePos) -> i64 {
        // Synthesized nodes (e.g. from JSX desugaring) carry `DUMMY_SP`
        // (`BytePos(0)`), below `file_start`; clamp instead of underflowing.
        let byte = bp.0.saturating_sub(self.file_start.0) as usize;
        match self.src.get(..byte) {
            Some(prefix) => prefix.encode_utf16().count() as i64,
            None => byte as i64,
        }
    }
    /// Resolve a name used in `scope` to the uid of the nearest enclosing scope
    /// that declares it (its `defScopeUid`), or `None` if global/undeclared.
    fn resolve(&self, name: &str, scope: u32) -> Option<u32> {
        let mut s = scope;
        loop {
            if self.decls.get(&s).is_some_and(|d| d.contains(name)) {
                return Some(s);
            }
            match self.parent.get(&s) {
                Some(&p) => s = p,
                None => return None,
            }
        }
    }
}

/// UTF-16 offsets where each line starts, using Babel's line-terminator set
/// (`\n`, `\r`, `\r\n`, `U+2028`, `U+2029`); `\r\n` counts as a single break.
fn compute_line_starts(src: &str) -> Vec<u32> {
    let mut starts = vec![0u32];
    let mut u16 = 0u32;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                u16 += 1;
                if chars.peek() == Some(&'\n') {
                    chars.next();
                    u16 += 1;
                }
                starts.push(u16);
            }
            '\n' | '\u{2028}' | '\u{2029}' => {
                u16 += c.len_utf16() as u32;
                starts.push(u16);
            }
            _ => u16 += c.len_utf16() as u32,
        }
    }
    starts
}

/// Assigns Babel-compatible `scopeUid`s by a pre-order walk that mirrors Babel's
/// scope-creation order: `Program` is scope 0, and each scope-creating node
/// (functions, non-body blocks, loops, `catch`, `switch`, classes) takes the
/// next uid as it is entered. Each identifier's span is recorded with the scope
/// it resolves to (the current scope), which is what the IR's identifier
/// attributes render.
#[derive(Default)]
struct ScopeAssigner {
    counter: u32,
    cur: u32,
    /// The nearest enclosing function/program scope. `var` bindings hoist here
    /// regardless of how many blocks they nest inside, so their `defScopeUid`
    /// is this rather than the textual scope.
    fn_scope: u32,
    /// A `catch`/function body block shares its parent's scope rather than
    /// opening a new one (`isScope(BlockStatement, Function|CatchClause)` is
    /// false in Babel); this flag tells the next block to not open a scope.
    skip_block: bool,
    map: HashMap<(u32, u32), u32>,
    /// `var` binding ident span -> hoisted (function) defScopeUid.
    var_defs: HashMap<(u32, u32), u32>,
    /// Scope tree: scope uid -> parent scope uid (program scope 0 has none).
    parent: HashMap<u32, u32>,
    /// Names declared in each scope (for resolving `referencedSymbol`).
    decls: HashMap<u32, std::collections::HashSet<String>>,
}

impl ScopeAssigner {
    fn fresh(&mut self) -> u32 {
        self.counter += 1;
        self.counter
    }
    /// Visit `n`'s children inside a freshly-created scope.
    fn scoped(&mut self, n: &impl VisitWith<Self>) {
        self.skip_block = false;
        let s = self.fresh();
        let saved = self.cur;
        self.parent.insert(s, saved);
        self.cur = s;
        n.visit_children_with(self);
        self.cur = saved;
    }
    /// Visit a function parameter pattern, opening a scope only when Babel does:
    /// a destructuring pattern param (`Array`/`Object`/`Assignment`) directly
    /// under a function is a scope, but a plain identifier and a `RestElement`
    /// (`...x` / `...[x]`) are NOT — `RestElement` is neither a `Pattern` nor
    /// `Scopable`, and its inner pattern's parent is the rest, not the function.
    fn visit_param_pat(&mut self, pat: &ast::Pat) {
        match pat {
            ast::Pat::Ident(_) => pat.visit_with(self),
            ast::Pat::Rest(r) => r.arg.visit_with(self),
            _ => self.scoped(pat),
        }
    }
    /// Record every binding ident span in a `var` pattern as hoisting to
    /// `scope` (the enclosing function/program scope).
    fn record_var_binding_spans(
        pat: &ast::Pat,
        scope: u32,
        out: &mut HashMap<(u32, u32), u32>,
    ) {
        match pat {
            ast::Pat::Ident(b) => {
                out.insert((b.id.span.lo.0, b.id.span.hi.0), scope);
            }
            ast::Pat::Array(a) => {
                for el in a.elems.iter().flatten() {
                    Self::record_var_binding_spans(el, scope, out);
                }
            }
            ast::Pat::Object(o) => {
                for p in &o.props {
                    match p {
                        ast::ObjectPatProp::KeyValue(kv) => {
                            Self::record_var_binding_spans(&kv.value, scope, out)
                        }
                        ast::ObjectPatProp::Assign(a) => {
                            out.insert((a.key.span().lo.0, a.key.span().hi.0), scope);
                        }
                        ast::ObjectPatProp::Rest(r) => {
                            Self::record_var_binding_spans(&r.arg, scope, out)
                        }
                    }
                }
            }
            ast::Pat::Assign(a) => Self::record_var_binding_spans(&a.left, scope, out),
            ast::Pat::Rest(r) => Self::record_var_binding_spans(&r.arg, scope, out),
            ast::Pat::Expr(_) | ast::Pat::Invalid(_) => {}
        }
    }
    fn record(&mut self, span: Span) {
        self.map.insert((span.lo.0, span.hi.0), self.cur);
    }
    /// Declare `name` in `scope` (for later `referencedSymbol` resolution).
    fn bind(&mut self, scope: u32, name: &str) {
        self.decls.entry(scope).or_default().insert(name.to_string());
    }
    /// Bind every identifier in a (possibly destructuring) pattern in `scope`.
    fn bind_pat(&mut self, pat: &ast::Pat, scope: u32) {
        match pat {
            ast::Pat::Ident(b) => self.bind(scope, b.id.sym.as_str()),
            ast::Pat::Array(a) => {
                for el in a.elems.iter().flatten() {
                    self.bind_pat(el, scope);
                }
            }
            ast::Pat::Object(o) => {
                for p in &o.props {
                    match p {
                        ast::ObjectPatProp::KeyValue(kv) => self.bind_pat(&kv.value, scope),
                        ast::ObjectPatProp::Assign(a) => self.bind(scope, a.key.sym.as_str()),
                        ast::ObjectPatProp::Rest(r) => self.bind_pat(&r.arg, scope),
                    }
                }
            }
            ast::Pat::Assign(a) => self.bind_pat(&a.left, scope),
            ast::Pat::Rest(r) => self.bind_pat(&r.arg, scope),
            ast::Pat::Expr(_) | ast::Pat::Invalid(_) => {}
        }
    }
    /// Enter a function: its `id`, simple params, and body live in the function
    /// scope; a non-simple (pattern) param opens its own nested scope, matching
    /// Babel's `isScope(Pattern, Function) == true`.
    fn enter_fn(&mut self, ident: Option<&ast::Ident>, func: &ast::Function) {
        self.enter_fn_named(ident, func, false)
    }
    /// `id_binds_here` is true for function *expressions* (the name binds in the
    /// function's own scope); declarations bind their name in the enclosing scope.
    fn enter_fn_named(&mut self, ident: Option<&ast::Ident>, func: &ast::Function, id_binds_here: bool) {
        let s = self.fresh();
        let saved = self.cur;
        let saved_fn = self.fn_scope;
        self.parent.insert(s, saved);
        self.cur = s;
        self.fn_scope = s;
        if let Some(id) = ident {
            self.record(id.span);
            if id_binds_here {
                self.bind(s, id.sym.as_str());
            }
        }
        for p in &func.params {
            self.bind_pat(&p.pat, s);
            self.visit_param_pat(&p.pat);
        }
        if let Some(body) = &func.body {
            // Body statements share the function scope (no extra block scope).
            for stmt in &body.stmts {
                stmt.visit_with(self);
            }
        }
        self.cur = saved;
        self.fn_scope = saved_fn;
    }
    /// Enter a class: its `id` and member keys live in the class scope; each
    /// method body opens its own nested scope (handled by the method visitors).
    fn enter_class(&mut self, ident: Option<&ast::Ident>, class: &ast::Class) {
        let s = self.fresh();
        let saved = self.cur;
        self.parent.insert(s, saved);
        self.cur = s;
        if let Some(id) = ident {
            self.record(id.span);
            // The class name is visible inside the class body (its own scope).
            self.bind(s, id.sym.as_str());
        }
        class.visit_children_with(self);
        self.cur = saved;
    }
    /// Record a property/method key at the current (object/class) scope. String
    /// and numeric keys carry a scope too; computed keys evaluate in this scope.
    fn record_prop_key(&mut self, key: &ast::PropName) {
        match key {
            ast::PropName::Ident(i) => self.record(i.span),
            ast::PropName::Str(s) => self.record(s.span),
            ast::PropName::Num(n) => self.record(n.span),
            ast::PropName::BigInt(b) => self.record(b.span),
            ast::PropName::Computed(c) => c.visit_with(self),
        }
    }
    /// Record a private name (`#x`) and its inner id (the name after `#`).
    fn record_private_name(&mut self, n: &ast::PrivateName) {
        self.record(n.span);
        let id = Span {
            lo: BytePos(n.span.lo.0 + 1),
            hi: n.span.hi,
        };
        self.map.insert((id.lo.0, id.hi.0), self.cur);
    }
}

impl Visit for ScopeAssigner {
    fn visit_ident(&mut self, n: &ast::Ident) {
        self.record(n.span);
    }
    fn visit_ident_name(&mut self, n: &ast::IdentName) {
        self.record(n.span);
    }
    fn visit_block_stmt(&mut self, n: &ast::BlockStmt) {
        if self.skip_block {
            self.skip_block = false;
            n.visit_children_with(self);
        } else {
            self.scoped(n);
        }
    }
    fn visit_while_stmt(&mut self, n: &ast::WhileStmt) {
        self.scoped(n);
    }
    fn visit_do_while_stmt(&mut self, n: &ast::DoWhileStmt) {
        self.scoped(n);
    }
    fn visit_for_stmt(&mut self, n: &ast::ForStmt) {
        self.scoped(n);
    }
    fn visit_for_in_stmt(&mut self, n: &ast::ForInStmt) {
        self.scoped(n);
    }
    fn visit_for_of_stmt(&mut self, n: &ast::ForOfStmt) {
        self.scoped(n);
    }
    fn visit_var_decl(&mut self, n: &ast::VarDecl) {
        // Record the declaration/declarator scopes (for-in/of declarations render
        // them); identifiers inside are recorded by `visit_ident`.
        self.record(n.span);
        // `var` bindings hoist to the enclosing function/program scope, so their
        // defScopeUid is `fn_scope` rather than the textual block scope.
        // `var` hoists to the function scope; `let`/`const` bind in the current
        // (block) scope.
        let bind_scope = if matches!(n.kind, ast::VarDeclKind::Var) {
            let fns = self.fn_scope;
            for d in &n.decls {
                Self::record_var_binding_spans(&d.name, fns, &mut self.var_defs);
            }
            fns
        } else {
            self.cur
        };
        for d in &n.decls {
            self.bind_pat(&d.name, bind_scope);
        }
        n.visit_children_with(self);
    }
    fn visit_var_declarator(&mut self, n: &ast::VarDeclarator) {
        self.record(n.span);
        n.visit_children_with(self);
    }
    fn visit_switch_stmt(&mut self, n: &ast::SwitchStmt) {
        self.scoped(n);
    }
    fn visit_catch_clause(&mut self, n: &ast::CatchClause) {
        // `catch (e)` binds `e` in the catch scope; a destructuring param
        // (`catch ({ x })`) is its own scope (Babel `isScope(Pattern, Catch)`).
        // The body block shares the catch scope.
        let s = self.fresh();
        let saved = self.cur;
        self.parent.insert(s, saved);
        self.cur = s;
        if let Some(p) = &n.param {
            self.bind_pat(p, s);
            self.visit_param_pat(p);
        }
        self.skip_block = true;
        n.body.visit_with(self);
        self.skip_block = false;
        self.cur = saved;
    }
    fn visit_import_decl(&mut self, n: &ast::ImportDecl) {
        // Import bindings live in the module (current) scope.
        for spec in &n.specifiers {
            let local = match spec {
                ast::ImportSpecifier::Named(s) => &s.local,
                ast::ImportSpecifier::Default(s) => &s.local,
                ast::ImportSpecifier::Namespace(s) => &s.local,
            };
            self.bind(self.cur, local.sym.as_str());
        }
        n.visit_children_with(self);
    }
    fn visit_fn_decl(&mut self, n: &ast::FnDecl) {
        // A function declaration's name binds in the enclosing scope.
        self.bind(self.cur, n.ident.sym.as_str());
        self.enter_fn(Some(&n.ident), &n.function);
    }
    fn visit_fn_expr(&mut self, n: &ast::FnExpr) {
        // A named function expression's name binds in its own scope.
        self.enter_fn_named(n.ident.as_ref(), &n.function, true);
    }
    fn visit_arrow_expr(&mut self, n: &ast::ArrowExpr) {
        // Like `enter_fn`: the arrow opens a function scope; a non-simple
        // (pattern) param opens its own nested scope, matching Babel's
        // `isScope(Pattern, Function) == true`. The body block shares the
        // arrow scope.
        let s = self.fresh();
        let saved = self.cur;
        let saved_fn = self.fn_scope;
        self.parent.insert(s, saved);
        self.cur = s;
        self.fn_scope = s;
        for p in &n.params {
            self.bind_pat(p, s);
            self.visit_param_pat(p);
        }
        match &*n.body {
            ast::BlockStmtOrExpr::BlockStmt(b) => {
                for stmt in &b.stmts {
                    stmt.visit_with(self);
                }
            }
            ast::BlockStmtOrExpr::Expr(e) => e.visit_with(self),
        }
        self.cur = saved;
        self.fn_scope = saved_fn;
    }
    fn visit_class_decl(&mut self, n: &ast::ClassDecl) {
        // A class declaration's name binds in the enclosing scope (and the class
        // scope, via `enter_class`).
        self.bind(self.cur, n.ident.sym.as_str());
        self.enter_class(Some(&n.ident), &n.class);
    }
    fn visit_class_expr(&mut self, n: &ast::ClassExpr) {
        self.enter_class(n.ident.as_ref(), &n.class);
    }
    fn visit_class_method(&mut self, n: &ast::ClassMethod) {
        self.record_prop_key(&n.key);
        self.enter_fn(None, &n.function);
    }
    fn visit_private_method(&mut self, n: &ast::PrivateMethod) {
        self.record_private_name(&n.key);
        self.enter_fn(None, &n.function);
    }
    fn visit_constructor(&mut self, n: &ast::Constructor) {
        // Key lives in the class scope; params + body open the ctor scope.
        self.record_prop_key(&n.key);
        let s = self.fresh();
        let saved = self.cur;
        let saved_fn = self.fn_scope;
        self.parent.insert(s, saved);
        self.cur = s;
        self.fn_scope = s;
        for p in &n.params {
            if let ast::ParamOrTsParamProp::Param(param) = p {
                self.bind_pat(&param.pat, s);
                self.visit_param_pat(&param.pat);
            }
        }
        if let Some(body) = &n.body {
            for stmt in &body.stmts {
                stmt.visit_with(self);
            }
        }
        self.cur = saved;
        self.fn_scope = saved_fn;
    }
    fn visit_class_prop(&mut self, n: &ast::ClassProp) {
        self.record_prop_key(&n.key);
        n.value.visit_with(self);
    }
    fn visit_private_prop(&mut self, n: &ast::PrivateProp) {
        self.record_private_name(&n.key);
        n.value.visit_with(self);
    }
    /// Private names in *use* position (`this.#x`, `#x in obj`) get the current
    /// scope's uid. Declaration keys are recorded explicitly above and their
    /// bodies are entered without re-descending the key, so this does not
    /// double-record them.
    fn visit_private_name(&mut self, n: &ast::PrivateName) {
        self.record_private_name(n);
    }
    fn visit_meta_prop_expr(&mut self, n: &ast::MetaPropExpr) {
        // `new.target` / `import.meta` synthesize two identifiers (meta + prop);
        // record their spans (matching `meta_prop_field`) at the current scope.
        let meta_len = match n.kind {
            ast::MetaPropKind::ImportMeta => "import".len() as u32,
            ast::MetaPropKind::NewTarget => "new".len() as u32,
        };
        let lo = n.span.lo.0;
        self.map.insert((lo, lo + meta_len), self.cur);
        self.map.insert((lo + meta_len + 1, n.span.hi.0), self.cur);
    }
    fn visit_method_prop(&mut self, n: &ast::MethodProp) {
        // The key is evaluated in the enclosing (object) scope; the body opens
        // its own method scope.
        self.record_prop_key(&n.key);
        self.enter_fn(None, &n.function);
    }
    fn visit_key_value_prop(&mut self, n: &ast::KeyValueProp) {
        // `{ "k": v }` / `{ 0: v }`: a string/numeric key carries a scopeUid too,
        // recorded here (the default visitor only records identifier keys).
        self.record_prop_key(&n.key);
        n.value.visit_with(self);
    }
    fn visit_key_value_pat_prop(&mut self, n: &ast::KeyValuePatProp) {
        // Same for a destructuring key: `{ "k": pat }` / `{ 0: pat }`.
        self.record_prop_key(&n.key);
        n.value.visit_with(self);
    }
    fn visit_getter_prop(&mut self, n: &ast::GetterProp) {
        self.record_prop_key(&n.key);
        let s = self.fresh();
        let saved = self.cur;
        self.cur = s;
        if let Some(b) = &n.body {
            for stmt in &b.stmts {
                stmt.visit_with(self);
            }
        }
        self.cur = saved;
    }
    fn visit_setter_prop(&mut self, n: &ast::SetterProp) {
        self.record_prop_key(&n.key);
        let s = self.fresh();
        let saved = self.cur;
        self.cur = s;
        n.param.visit_with(self);
        if let Some(b) = &n.body {
            for stmt in &b.stmts {
                stmt.visit_with(self);
            }
        }
        self.cur = saved;
    }
}

/// A borrowed handle to one swc node (or synthesized sub-record), tagged with
/// which it is. `Copy`, so forming children is cheap; the bump arena owns the
/// wrappers we hand out as `&dyn AstNode`.
#[derive(Clone, Copy)]
pub struct W<'a> {
    node: Ref<'a>,
    bump: &'a Bump,
    cx: &'a Cx,
}

#[derive(Clone, Copy)]
enum Ref<'a> {
    File(&'a ast::Program),
    Program(&'a ast::Program),
    ModuleItem(&'a ast::ModuleItem),
    Stmt(&'a ast::Stmt),
    Expr(&'a ast::Expr),
    /// A swc `Ident`-like name without its own `Ident` node (e.g. member props).
    Name(&'a ast::IdentName),
    /// A bare swc `Ident` (e.g. a statement label).
    Ident(&'a ast::Ident),
    BindingIdent(&'a ast::BindingIdent),
    Pat(&'a ast::Pat),
    Member(&'a ast::MemberExpr),
    Paren(&'a ast::ParenExpr),
    /// `SpreadElement { argument }` wrapping the inner expression.
    Spread(&'a ast::Expr),
    VarDecl(&'a ast::VarDecl),
    VarDeclarator(&'a ast::VarDeclarator),
    /// A declaration inside `export <decl>` (`export function`/`class`/`var`).
    Decl(&'a ast::Decl),
    /// A declaration inside `export default <decl>` (`export default
    /// function`/`class`), which Babel models as a `FunctionDeclaration`/
    /// `ClassDeclaration` (with an optionally-null `id`).
    DefaultDecl(&'a ast::DefaultDecl),
    Block(&'a ast::BlockStmt),
    /// A function/method/arrow body block. Identical to `Block` except its
    /// leading string-literal statements are split off as Babel `directives`.
    FnBody(&'a ast::BlockStmt),
    CatchClause(&'a ast::CatchClause),
    SwitchCase(&'a ast::SwitchCase),
    TplElement(&'a ast::TplElement),
    /// A template element's `value` record (`cooked` + `raw`).
    TplValue(&'a ast::TplElement),
    /// A literal's `extra` record (`raw` + `rawValue`).
    NumExtra(&'a ast::Number),
    StrExtra(&'a ast::Str),
    /// A synthesized `SourceLocation` record (`start`/`end` positions +
    /// `identifierName`) for the given span; the name is set for identifiers.
    Loc(Span, Option<&'a str>),
    /// A synthesized `{ line, column }` position record.
    Pos(i64, i64),
    /// A source comment (`File.comments` entry).
    Comment(&'a Comment),
    /// A directive (`'use strict';`) wrapping its string literal.
    Directive(&'a ast::Str),
    /// A directive's `DirectiveLiteral` (value + extra), from the string literal.
    DirectiveLiteral(&'a ast::Str),
    /// The program's interpreter directive (`#!...`); swc keeps only the text,
    /// so the span is synthesized to cover `#!` plus the text from offset 0.
    Interpreter(&'a str),
    /// A bare template literal (the `quasi` of a tagged template).
    TplLit(&'a ast::Tpl),
    /// A regular-expression literal's `extra` record (`raw`).
    RegexExtra(&'a ast::Regex),
    /// A bigint literal's `extra` record (`raw` + `rawValue`).
    BigIntExtra(&'a ast::BigInt),
    /// A synthesized identifier with an explicit span and name (the `import`/
    /// `meta` parts of a meta-property, or a private name's inner id), where swc
    /// has no standalone identifier node.
    SynthIdent(Span, &'a str),
    /// A class body, one of its members, and a private name (`#x`).
    ClassBody(&'a ast::Class),
    ClassMember(&'a ast::ClassMember),
    PrivateName(&'a ast::PrivateName),
    /// An object-literal member (`ObjectProperty` or `ObjectMethod`).
    ObjProp(&'a ast::Prop),
    /// A property key presented as a `StringLiteral` / `NumericLiteral` node.
    StrLit(&'a ast::Str),
    NumLit(&'a ast::Number),
    /// An object destructuring pattern and one of its properties.
    ObjectPat(&'a ast::ObjectPat),
    /// An array destructuring pattern in assignment-target position (`[a] = b`).
    ArrayPat(&'a ast::ArrayPat),
    ObjPatProp(&'a ast::ObjectPatProp),
    /// The synthetic `AssignmentPattern` for `{ x = default }` shorthand: swc
    /// keeps key+default on one `AssignPatProp`, Babel splits it into an
    /// `ObjectProperty` whose value is an `AssignmentPattern`.
    AssignPatPropDefault(&'a ast::AssignPatProp),
    /// A `definedSymbols` entry: a binding's `name` + `defScopeUid`.
    DefinedSymbol(&'a str, i64),
    ImportSpec(&'a ast::ImportSpecifier),
    ExportSpec(&'a ast::ExportSpecifier),
    /// The `import` keyword as a dynamic-import callee (`import(x)`).
    Import(&'a ast::Import),
    /// The `super` keyword as a call/member callee (a leaf `Super` node).
    Super(&'a ast::Super),
    /// `super.x` / `super[x]`, which Babel models as a `MemberExpression` whose
    /// `object` is a `Super` node.
    SuperProp(&'a ast::SuperPropExpr),
}

impl<'a> W<'a> {
    fn wrap(&self, r: Ref<'a>) -> &'a dyn AstNode {
        self.bump.alloc(W {
            node: r,
            bump: self.bump,
            cx: self.cx,
        })
    }

    /// The span of the wrapped swc node, or `None` for synthesized records.
    fn span(&self) -> Option<Span> {
        Some(match self.node {
            Ref::File(p) | Ref::Program(p) => p.span(),
            Ref::ModuleItem(m) => m.span(),
            Ref::Stmt(s) => s.span(),
            Ref::Expr(e) => e.span(),
            Ref::Name(n) => n.span(),
            Ref::Ident(i) => i.span(),
            Ref::BindingIdent(b) => b.span(),
            Ref::Pat(p) => p.span(),
            Ref::Member(m) => m.span(),
            Ref::Paren(p) => p.span(),
            Ref::Spread(e) => e.span(),
            Ref::VarDecl(v) => v.span(),
            Ref::VarDeclarator(d) => d.span(),
            Ref::Decl(d) => d.span(),
            Ref::DefaultDecl(d) => match d {
                ast::DefaultDecl::Fn(f) => f.function.span(),
                ast::DefaultDecl::Class(c) => c.class.span(),
                ast::DefaultDecl::TsInterfaceDecl(i) => i.span(),
            },
            Ref::Block(b) | Ref::FnBody(b) => b.span(),
            Ref::CatchClause(c) => c.span(),
            Ref::SwitchCase(c) => c.span(),
            Ref::TplElement(t) => t.span(),
            Ref::Comment(c) => c.span,
            Ref::Directive(s) | Ref::DirectiveLiteral(s) => s.span,
            Ref::SynthIdent(span, _) => span,
            Ref::ObjProp(p) => p.span(),
            Ref::StrLit(s) => s.span,
            Ref::NumLit(n) => n.span,
            Ref::ClassBody(c) => c.span,
            Ref::ClassMember(m) => m.span(),
            Ref::PrivateName(p) => p.span,
            Ref::ObjectPat(o) => o.span,
            Ref::ArrayPat(a) => a.span,
            Ref::ObjPatProp(p) => p.span(),
            Ref::AssignPatPropDefault(p) => p.span,
            Ref::Import(i) => i.span,
            Ref::Super(s) => s.span,
            Ref::SuperProp(m) => m.span,
                        Ref::ImportSpec(s) => s.span(),
            Ref::ExportSpec(s) => s.span(),
            Ref::Interpreter(text) => {
                // `#!` (2 bytes) + the interpreter text, anchored at file start.
                let lo = self.cx.file_start;
                Span {
                    lo,
                    hi: BytePos(lo.0 + 2 + text.len() as u32),
                }
            }
            Ref::TplLit(t) => t.span(),
            Ref::TplValue(_)
            | Ref::NumExtra(_)
            | Ref::StrExtra(_)
            | Ref::RegexExtra(_)
            | Ref::BigIntExtra(_)
            | Ref::DefinedSymbol(..)
            | Ref::Loc(..)
            | Ref::Pos(..) => return None,
        })
    }

    /// The identifier name to record in `loc.identifierName`, for identifiers.
    fn ident_name(&self) -> Option<&'a str> {
        match self.node {
            Ref::Name(i) => Some(i.sym.as_str()),
            Ref::Ident(i) => Some(i.sym.as_str()),
            Ref::BindingIdent(b) => Some(b.id.sym.as_str()),
            Ref::SynthIdent(_, name) => Some(name),
            Ref::Expr(ast::Expr::Ident(i)) => Some(i.sym.as_str()),
            _ => None,
        }
    }

    /// `loc`: a `SourceLocation` record, when this node has a span.
    fn loc_field(&self) -> Field<'a> {
        match self.span() {
            Some(s) => self.node(Ref::Loc(s, self.ident_name())),
            None => Field::Absent,
        }
    }

    /// `scopeUid`: present on every node from `Program` down (not on `File`).
    /// Identifiers look up their Babel scope from the precomputed map; other
    /// nodes report the program scope (0) since only identifier attributes
    /// render a scope.
    fn scope_field(&self) -> Field<'a> {
        match self.node {
            Ref::File(_) => Field::Absent,
            _ => {
                let uid = self
                    .span()
                    .and_then(|s| self.cx.scopes.get(&(s.lo.0, s.hi.0)).copied())
                    .unwrap_or(0);
                Field::Int(uid as i64)
            }
        }
    }
    /// `referencedSymbol`: for an identifier, the `{name, defScopeUid}` of the
    /// binding it resolves to (walking the scope chain). Absent for non-idents
    /// and unresolved (global/undeclared) names. This lives only in the IR
    /// trivia (the generic printer elides it), so it never affects `ast2hir`
    /// byte-exactness; it feeds the dataflow analyses' per-variable state.
    fn referenced_symbol_field(&self) -> Field<'a> {
        let Some(name) = self.ident_name() else {
            return Field::Absent;
        };
        let scope = self
            .span()
            .and_then(|s| self.cx.scopes.get(&(s.lo.0, s.hi.0)).copied())
            .unwrap_or(0);
        match self.cx.resolve(name, scope) {
            Some(def_scope) => self.node(Ref::DefinedSymbol(name, def_scope as i64)),
            None => Field::Absent,
        }
    }
    fn node(&self, r: Ref<'a>) -> Field<'a> {
        Field::Node(self.wrap(r))
    }
    fn expr(&self, e: &'a ast::Expr) -> Field<'a> {
        self.node(Ref::Expr(e))
    }
    fn stmt_list(&self, body: impl Iterator<Item = &'a ast::Stmt>) -> Field<'a> {
        Field::List(body.map(|s| self.node(Ref::Stmt(s))).collect())
    }
    /// A call/new argument or array element: a spread becomes a `SpreadElement`.
    fn arg(&self, a: &'a ast::ExprOrSpread) -> Field<'a> {
        if a.spread.is_some() {
            self.node(Ref::Spread(&a.expr))
        } else {
            self.expr(&a.expr)
        }
    }
    /// An assignment target (`a`, `a.b`, `(a)`) presented as the matching node.
    fn assign_target(&self, t: &'a ast::AssignTarget) -> Field<'a> {
        match t {
            ast::AssignTarget::Simple(s) => match s {
                ast::SimpleAssignTarget::Ident(b) => self.node(Ref::BindingIdent(b)),
                ast::SimpleAssignTarget::Member(m) => self.node(Ref::Member(m)),
                ast::SimpleAssignTarget::SuperProp(m) => self.node(Ref::SuperProp(m)),
                ast::SimpleAssignTarget::Paren(p) => self.node(Ref::Paren(p)),
                _ => Field::Absent,
            },
            ast::AssignTarget::Pat(p) => match p {
                ast::AssignTargetPat::Object(o) => self.node(Ref::ObjectPat(o)),
                ast::AssignTargetPat::Array(a) => self.node(Ref::ArrayPat(a)),
                _ => Field::Absent,
            },
        }
    }
    /// A binding pattern (`a`, or a destructuring pattern) as a node.
    fn pat(&self, p: &'a ast::Pat) -> Field<'a> {
        match p {
            ast::Pat::Ident(b) => self.node(Ref::BindingIdent(b)),
            ast::Pat::Expr(e) => self.expr(e),
            _ => self.node(Ref::Pat(p)),
        }
    }

    /// Array-pattern elements, reconstructing the **trailing elisions** that swc
    /// drops in pattern position (`[a, ,]` parses to `[Some(a)]`, `[,]` to `[]`)
    /// but Babel keeps as holes. The dropped holes are recovered by counting the
    /// commas between the last parsed element and the closing `]` (that tail
    /// contains only commas/whitespace, so the count is unambiguous).
    fn array_pat_elements(&self, a: &'a ast::ArrayPat) -> Vec<Field<'a>> {
        let mut out: Vec<Field<'a>> = a
            .elems
            .iter()
            .map(|el| match el {
                Some(p) => self.pat(p),
                None => Field::Null,
            })
            .collect();
        let last_end = a.elems.iter().flatten().map(|p| p.span().hi.0).max().unwrap_or(a.span.lo.0 + 1);
        let from = (last_end - self.cx.file_start.0) as usize;
        let to = (a.span.hi.0 - self.cx.file_start.0) as usize;
        let commas = self.cx.src.get(from..to).map(|s| s.bytes().filter(|&b| b == b',').count()).unwrap_or(0);
        // With ≥1 parsed element the first comma is its separator; with none,
        // every comma up to the closing `]` is a hole.
        let trailing = if a.elems.is_empty() { commas } else { commas.saturating_sub(1) };
        out.extend(std::iter::repeat_with(|| Field::Null).take(trailing));
        out
    }

    fn pat_field(&self, p: &'a ast::Pat, key: &str) -> Field<'a> {
        match p {
            ast::Pat::Object(o) => self.object_pat_field(o, key),
            ast::Pat::Array(a) if key == "elements" => Field::List(self.array_pat_elements(a)),
            ast::Pat::Rest(r) if key == "argument" => self.pat(&r.arg),
            ast::Pat::Assign(a) => match key {
                "left" => self.pat(&a.left),
                "right" => self.expr(&a.right),
                _ => Field::Absent,
            },
            _ => Field::Absent,
        }
    }
}

impl<'a> AstNode for W<'a> {
    fn node_type(&self) -> &str {
        match self.node {
            Ref::File(_) => "File",
            Ref::Program(_) => "Program",
            Ref::ModuleItem(ast::ModuleItem::Stmt(s)) => stmt_type(s),
            Ref::ModuleItem(ast::ModuleItem::ModuleDecl(m)) => {
                module_decl_type(m)
            }
            Ref::ImportSpec(s) => import_spec_type(s),
            Ref::ExportSpec(_) => "ExportSpecifier",
            Ref::Stmt(s) => stmt_type(s),
            Ref::Expr(e) => expr_type(e),
            Ref::Name(_) | Ref::Ident(_) | Ref::BindingIdent(_) | Ref::SynthIdent(..) => {
                "Identifier"
            }
            Ref::Pat(p) => pat_type(p),
            Ref::Member(_) => "MemberExpression",
            Ref::Paren(_) => "ParenthesizedExpression",
            Ref::Spread(_) => "SpreadElement",
            Ref::VarDecl(_) => "VariableDeclaration",
            Ref::VarDeclarator(_) => "VariableDeclarator",
            Ref::Decl(ast::Decl::Fn(_)) => "FunctionDeclaration",
            Ref::Decl(ast::Decl::Class(_)) => "ClassDeclaration",
            Ref::Decl(ast::Decl::Var(_)) => "VariableDeclaration",
            Ref::Decl(_) => "FunctionDeclaration",
            Ref::DefaultDecl(ast::DefaultDecl::Class(_)) => "ClassDeclaration",
            Ref::DefaultDecl(_) => "FunctionDeclaration",
            Ref::Block(_) | Ref::FnBody(_) => "BlockStatement",
            Ref::CatchClause(_) => "CatchClause",
            Ref::SwitchCase(_) => "SwitchCase",
            Ref::TplElement(_) => "TemplateElement",
            // Record names are unused by `ast2hir` (it reads fields directly).
            Ref::TplValue(_) | Ref::NumExtra(_) | Ref::StrExtra(_) => "Extra",
            Ref::Loc(..) => "SourceLocation",
            Ref::Pos(..) => "Position",
            Ref::Comment(c) => match c.kind {
                CommentKind::Line => "CommentLine",
                CommentKind::Block => "CommentBlock",
            },
            Ref::Directive(_) => "Directive",
            Ref::DirectiveLiteral(_) => "DirectiveLiteral",
            Ref::Interpreter(_) => "InterpreterDirective",
            Ref::TplLit(_) => "TemplateLiteral",
            Ref::RegexExtra(_) | Ref::BigIntExtra(_) => "Extra",
            Ref::ObjProp(p) => obj_prop_type(p),
            Ref::StrLit(_) => "StringLiteral",
            Ref::NumLit(_) => "NumericLiteral",
            Ref::ClassBody(_) => "ClassBody",
            Ref::ClassMember(m) => class_member_type(m),
            Ref::PrivateName(_) => "PrivateName",
            Ref::ObjectPat(_) => "ObjectPattern",
            Ref::ArrayPat(_) => "ArrayPattern",
            Ref::ObjPatProp(ast::ObjectPatProp::Rest(_)) => "RestElement",
            Ref::ObjPatProp(_) => "ObjectProperty",
            Ref::AssignPatPropDefault(_) => "AssignmentPattern",
            Ref::Import(_) => "Import",
            Ref::Super(_) => "Super",
            Ref::SuperProp(_) => "MemberExpression",
            Ref::DefinedSymbol(..) => "Symbol",
        }
    }

    fn field(&self, key: &str) -> Field<'_> {
        // Synthesized records carry their own (non-span) fields; handle them
        // before the base-trivia intercept (their `start`/`end` are positions,
        // not source offsets).
        match self.node {
            Ref::Loc(span, name) => {
                return match key {
                    "start" => {
                        let (l, c) = self.cx.line_col(span.lo);
                        self.node(Ref::Pos(l, c))
                    }
                    "end" => {
                        let (l, c) = self.cx.line_col(span.hi);
                        self.node(Ref::Pos(l, c))
                    }
                    "identifierName" => name.map(Field::Str).unwrap_or(Field::Absent),
                    _ => Field::Absent,
                }
            }
            Ref::Pos(line, column) => {
                return match key {
                    "line" => Field::Int(line),
                    "column" => Field::Int(column),
                    _ => Field::Absent,
                }
            }
            Ref::TplValue(t) => return tpl_value_field(t, key),
            Ref::NumExtra(n) => return self.num_extra_field(n, key),
            Ref::StrExtra(s) => return str_extra_field(s, key),
            _ => {}
        }
        // Base trivia, shared by every real node.
        match key {
            "loc" => return self.loc_field(),
            "start" => {
                return self
                    .span()
                    .map(|s| Field::Int(self.cx.offset(s.lo)))
                    .unwrap_or(Field::Absent)
            }
            "end" => {
                return self
                    .span()
                    .map(|s| Field::Int(self.cx.offset(s.hi)))
                    .unwrap_or(Field::Absent)
            }
            "scopeUid" => return self.scope_field(),
            "referencedSymbol" => return self.referenced_symbol_field(),
            _ => {}
        }
        match self.node {
            Ref::File(p) => match key {
                "program" => self.node(Ref::Program(p)),
                "comments" => Field::List(
                    self.cx
                        .comments
                        .iter()
                        .map(|c| self.node(Ref::Comment(c)))
                        .collect(),
                ),
                _ => Field::Absent,
            },
            Ref::Program(p) => self.program_field(p, key),
            Ref::ModuleItem(ast::ModuleItem::Stmt(s)) => self.stmt_field(s, key),
            Ref::ModuleItem(ast::ModuleItem::ModuleDecl(m)) => {
                self.module_decl_field(m, key)
            }
            Ref::ImportSpec(s) => self.import_spec_field(s, key),
            Ref::ExportSpec(s) => self.export_spec_field(s, key),
            Ref::Stmt(s) => self.stmt_field(s, key),
            Ref::Expr(e) => self.expr_field(e, key),
            Ref::Name(i) if key == "name" => Field::Str(i.sym.as_str()),
            Ref::Ident(i) if key == "name" => Field::Str(i.sym.as_str()),
            Ref::SynthIdent(_, name) if key == "name" => Field::Str(name),
            Ref::BindingIdent(b) if key == "name" => Field::Str(b.id.sym.as_str()),
            Ref::Pat(p) => self.pat_field(p, key),
            Ref::Member(m) => self.member_field(m, key),
            Ref::SuperProp(m) => self.super_prop_field(m, key),
            Ref::Paren(p) if key == "expression" => self.expr(&p.expr),
            Ref::Spread(e) if key == "argument" => self.expr(e),
            Ref::VarDecl(v) => self.var_decl_field(v, key),
            Ref::VarDeclarator(d) => self.declarator_field(d, key),
            Ref::Decl(ast::Decl::Var(v)) => self.var_decl_field(v, key),
            Ref::Decl(ast::Decl::Fn(f)) => self.function_field(Some(&f.ident), &f.function, key),
            Ref::Decl(ast::Decl::Class(c)) => self.class_field(Some(&c.ident), &c.class, key),
            Ref::Decl(_) => Field::Absent,
            Ref::DefaultDecl(ast::DefaultDecl::Fn(f)) => {
                self.function_field(f.ident.as_ref(), &f.function, key)
            }
            Ref::DefaultDecl(ast::DefaultDecl::Class(c)) => {
                self.class_field(c.ident.as_ref(), &c.class, key)
            }
            Ref::DefaultDecl(_) => Field::Absent,
            Ref::Block(b) if key == "body" => self.stmt_list(b.stmts.iter()),
            // A function body splits its leading string-literal prologue into
            // Babel `directives`; the rest is `body`.
            Ref::FnBody(b) => match key {
                "directives" => {
                    let (dirs, _) = split_directives(&b.stmts);
                    Field::List(dirs.into_iter().map(|d| self.node(Ref::Directive(d))).collect())
                }
                "body" => {
                    let (_, rest) = split_directives(&b.stmts);
                    self.stmt_list(rest.iter())
                }
                _ => Field::Absent,
            },
            Ref::CatchClause(c) => match key {
                "param" => match &c.param {
                    Some(p) => self.pat(p),
                    None => Field::Null,
                },
                "body" => self.node(Ref::Block(&c.body)),
                _ => Field::Absent,
            },
            Ref::SwitchCase(c) => match key {
                "test" => match &c.test {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "consequent" => self.stmt_list(c.cons.iter()),
                _ => Field::Absent,
            },
            Ref::TplElement(t) => match key {
                "tail" => Field::Bool(t.tail),
                "value" => self.node(Ref::TplValue(t)),
                _ => Field::Absent,
            },
            Ref::Comment(c) if key == "value" => Field::Str(&c.text),
            Ref::Directive(s) if key == "value" => self.node(Ref::DirectiveLiteral(s)),
            Ref::DirectiveLiteral(s) => match key {
                "value" => str_value(s),
                "extra" => self.node(Ref::StrExtra(s)),
                _ => Field::Absent,
            },
            Ref::Interpreter(text) if key == "value" => Field::Str(text),
            Ref::TplLit(t) => self.tpl_field(t, key),
            Ref::RegexExtra(r) if key == "raw" => {
                Field::Str(self.bump.alloc_str(&format!("/{}/{}", r.exp, r.flags)))
            }
            Ref::BigIntExtra(n) => match key {
                "raw" => match n.raw.as_deref() {
                    Some(r) => Field::Str(r),
                    None => Field::Absent,
                },
                "rawValue" => self.bigint_value(n),
                _ => Field::Absent,
            },
            Ref::ObjProp(p) => self.obj_prop_field(p, key),
            Ref::StrLit(s) => match key {
                "value" => str_value(s),
                "extra" => self.node(Ref::StrExtra(s)),
                _ => Field::Absent,
            },
            Ref::NumLit(n) => match key {
                "value" => Field::Float(n.value),
                "extra" => self.node(Ref::NumExtra(n)),
                _ => Field::Absent,
            },
            Ref::ObjectPat(o) => self.object_pat_field(o, key),
            Ref::ArrayPat(a) if key == "elements" => Field::List(self.array_pat_elements(a)),
            Ref::ObjPatProp(p) => self.obj_pat_prop_field(p, key),
            Ref::AssignPatPropDefault(p) => match key {
                "left" => self.node(Ref::BindingIdent(&p.key)),
                "right" => match &p.value {
                    Some(e) => self.expr(e),
                    None => Field::Absent,
                },
                _ => Field::Absent,
            },
            Ref::ClassBody(c) if key == "body" => Field::List(
                c.body
                    .iter()
                    .filter(|m| !matches!(m, ast::ClassMember::Empty(_)))
                    .map(|m| self.node(Ref::ClassMember(m)))
                    .collect(),
            ),
            Ref::ClassMember(m) => self.class_member_field(m, key),
            Ref::PrivateName(p) => match key {
                // The inner id is the name without the leading `#`.
                "id" => {
                    let id_span = Span {
                        lo: BytePos(p.span.lo.0 + 1),
                        hi: p.span.hi,
                    };
                    self.node(Ref::SynthIdent(id_span, &p.name))
                }
                _ => Field::Absent,
            },
            Ref::DefinedSymbol(name, scope) => match key {
                "name" => Field::Str(name),
                "defScopeUid" => Field::Int(scope),
                _ => Field::Absent,
            },
            _ => Field::Absent,
        }
    }
}

fn module_decl_type(m: &ast::ModuleDecl) -> &'static str {
    match m {
        ast::ModuleDecl::Import(_) => "ImportDeclaration",
        ast::ModuleDecl::ExportDecl(_) | ast::ModuleDecl::ExportNamed(_) => {
            "ExportNamedDeclaration"
        }
        ast::ModuleDecl::ExportDefaultDecl(_) | ast::ModuleDecl::ExportDefaultExpr(_) => {
            "ExportDefaultDeclaration"
        }
        ast::ModuleDecl::ExportAll(_) => "ExportAllDeclaration",
        _ => "UnsupportedModuleDecl",
    }
}

fn class_member_type(m: &ast::ClassMember) -> &'static str {
    match m {
        ast::ClassMember::ClassProp(_) => "ClassProperty",
        ast::ClassMember::PrivateProp(_) => "ClassPrivateProperty",
        ast::ClassMember::Method(_) => "ClassMethod",
        ast::ClassMember::PrivateMethod(_) => "ClassPrivateMethod",
        ast::ClassMember::Constructor(_) => "ClassMethod",
        _ => "UnsupportedClassMember",
    }
}

fn method_kind(k: ast::MethodKind) -> &'static str {
    match k {
        ast::MethodKind::Method => "method",
        ast::MethodKind::Getter => "get",
        ast::MethodKind::Setter => "set",
    }
}

fn import_spec_type(s: &ast::ImportSpecifier) -> &'static str {
    match s {
        ast::ImportSpecifier::Named(_) => "ImportSpecifier",
        ast::ImportSpecifier::Default(_) => "ImportDefaultSpecifier",
        ast::ImportSpecifier::Namespace(_) => "ImportNamespaceSpecifier",
    }
}

/// `ObjectProperty` for data properties / shorthand, `ObjectMethod` for methods.
fn obj_prop_type(p: &ast::Prop) -> &'static str {
    match p {
        ast::Prop::Shorthand(_) | ast::Prop::KeyValue(_) | ast::Prop::Assign(_) => "ObjectProperty",
        ast::Prop::Method(_) | ast::Prop::Getter(_) | ast::Prop::Setter(_) => "ObjectMethod",
    }
}

impl<'a> W<'a> {
    fn program_field(&self, p: &'a ast::Program, key: &str) -> Field<'a> {
        // The shebang is the interpreter directive; it lives only on the program.
        let shebang = match p {
            ast::Program::Script(s) => s.shebang.as_deref(),
            ast::Program::Module(m) => m.shebang.as_deref(),
        };
        match (p, key) {
            (_, "sourceType") => Field::Str(match p {
                ast::Program::Module(_) => "module",
                ast::Program::Script(_) => "script",
            }),
            (_, "interpreter") => match shebang {
                Some(text) => self.node(Ref::Interpreter(text)),
                None => Field::Null,
            },
            // Scripts carry a directive prologue; split it off from the body.
            (ast::Program::Script(s), "directives") => {
                let (dirs, _) = split_directives(&s.body);
                Field::List(dirs.into_iter().map(|d| self.node(Ref::Directive(d))).collect())
            }
            (ast::Program::Script(s), "body") => {
                let (_, rest) = split_directives(&s.body);
                self.stmt_list(rest.iter())
            }
            (ast::Program::Module(_), "directives") => Field::List(vec![]),
            (ast::Program::Module(m), "body") => Field::List(
                m.body
                    .iter()
                    .map(|it| self.node(Ref::ModuleItem(it)))
                    .collect(),
            ),
            _ => Field::Absent,
        }
    }

    fn stmt_field(&self, s: &'a ast::Stmt, key: &str) -> Field<'a> {
        match s {
            ast::Stmt::Expr(e) if key == "expression" => self.expr(&e.expr),
            ast::Stmt::Return(r) if key == "argument" => match &r.arg {
                Some(e) => self.expr(e),
                None => Field::Null,
            },
            ast::Stmt::Throw(t) if key == "argument" => self.expr(&t.arg),
            ast::Stmt::With(w) => match key {
                "object" => self.expr(&w.obj),
                "body" => self.node(Ref::Stmt(&w.body)),
                _ => Field::Absent,
            },
            ast::Stmt::Block(b) if key == "body" => self.stmt_list(b.stmts.iter()),
            ast::Stmt::If(i) => match key {
                "test" => self.expr(&i.test),
                "consequent" => self.node(Ref::Stmt(&i.cons)),
                "alternate" => match &i.alt {
                    Some(a) => self.node(Ref::Stmt(a)),
                    None => Field::Null,
                },
                _ => Field::Absent,
            },
            ast::Stmt::While(w) => match key {
                "test" => self.expr(&w.test),
                "body" => self.node(Ref::Stmt(&w.body)),
                _ => Field::Absent,
            },
            ast::Stmt::DoWhile(w) => match key {
                "test" => self.expr(&w.test),
                "body" => self.node(Ref::Stmt(&w.body)),
                _ => Field::Absent,
            },
            ast::Stmt::For(f) => match key {
                "init" => match &f.init {
                    Some(ast::VarDeclOrExpr::VarDecl(v)) => self.node(Ref::VarDecl(v)),
                    Some(ast::VarDeclOrExpr::Expr(e)) => self.expr(e),
                    None => Field::Null,
                },
                "test" => match &f.test {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "update" => match &f.update {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "body" => self.node(Ref::Stmt(&f.body)),
                _ => Field::Absent,
            },
            ast::Stmt::Try(t) => match key {
                "block" => self.node(Ref::Block(&t.block)),
                "handler" => match &t.handler {
                    Some(h) => self.node(Ref::CatchClause(h)),
                    None => Field::Null,
                },
                "finalizer" => match &t.finalizer {
                    Some(b) => self.node(Ref::Block(b)),
                    None => Field::Null,
                },
                _ => Field::Absent,
            },
            ast::Stmt::Switch(s) => match key {
                "discriminant" => self.expr(&s.discriminant),
                "cases" => Field::List(
                    s.cases
                        .iter()
                        .map(|c| self.node(Ref::SwitchCase(c)))
                        .collect(),
                ),
                _ => Field::Absent,
            },
            ast::Stmt::Labeled(l) => match key {
                "label" => self.node(Ref::Ident(&l.label)),
                "body" => self.node(Ref::Stmt(&l.body)),
                _ => Field::Absent,
            },
            ast::Stmt::Break(b) if key == "label" => match &b.label {
                Some(l) => self.node(Ref::Ident(l)),
                None => Field::Null,
            },
            ast::Stmt::Continue(c) if key == "label" => match &c.label {
                Some(l) => self.node(Ref::Ident(l)),
                None => Field::Null,
            },
            ast::Stmt::ForIn(f) => match key {
                "left" => self.for_head(&f.left),
                "right" => self.expr(&f.right),
                "body" => self.node(Ref::Stmt(&f.body)),
                _ => Field::Absent,
            },
            ast::Stmt::ForOf(f) => match key {
                "left" => self.for_head(&f.left),
                "right" => self.expr(&f.right),
                "body" => self.node(Ref::Stmt(&f.body)),
                "await" => Field::Bool(f.is_await),
                _ => Field::Absent,
            },
            ast::Stmt::Decl(ast::Decl::Var(v)) => self.var_decl_field(v, key),
            ast::Stmt::Decl(ast::Decl::Fn(f)) => {
                self.function_field(Some(&f.ident), &f.function, key)
            }
            ast::Stmt::Decl(ast::Decl::Class(c)) => {
                self.class_field(Some(&c.ident), &c.class, key)
            }
            _ => Field::Absent,
        }
    }

    /// The `left` of a for-in/of: a `let`/`const`/`var` declaration or an lval.
    fn for_head(&self, h: &'a ast::ForHead) -> Field<'a> {
        match h {
            ast::ForHead::VarDecl(v) => self.node(Ref::VarDecl(v)),
            ast::ForHead::Pat(p) => self.pat(p),
            ast::ForHead::UsingDecl(_) => Field::Absent,
        }
    }

    fn var_decl_field(&self, v: &'a ast::VarDecl, key: &str) -> Field<'a> {
        match key {
            "kind" => Field::Str(match v.kind {
                ast::VarDeclKind::Var => "var",
                ast::VarDeclKind::Let => "let",
                ast::VarDeclKind::Const => "const",
            }),
            "declarations" => Field::List(
                v.decls
                    .iter()
                    .map(|d| self.node(Ref::VarDeclarator(d)))
                    .collect(),
            ),
            _ => Field::Absent,
        }
    }

    fn declarator_field(&self, d: &'a ast::VarDeclarator, key: &str) -> Field<'a> {
        match key {
            "id" => self.pat(&d.name),
            "init" => match &d.init {
                Some(e) => self.expr(e),
                None => Field::Null,
            },
            // The bindings introduced by this declarator (used by for-in/of
            // declarations). A destructuring pattern binds several names, in
            // source order.
            "definedSymbols" => {
                let mut out = Vec::new();
                self.collect_pat_symbols(&d.name, &mut out);
                Field::List(out)
            }
            _ => Field::Absent,
        }
    }

    /// Collect every binding identifier introduced by a (possibly destructuring)
    /// pattern, in source order, as `DefinedSymbol` nodes.
    fn collect_pat_symbols(&self, pat: &'a ast::Pat, out: &mut Vec<Field<'a>>) {
        match pat {
            ast::Pat::Ident(b) => {
                let scope = self.def_scope_of(b.id.span);
                out.push(self.node(Ref::DefinedSymbol(b.id.sym.as_str(), scope)));
            }
            ast::Pat::Array(a) => {
                for el in a.elems.iter().flatten() {
                    self.collect_pat_symbols(el, out);
                }
            }
            ast::Pat::Object(o) => {
                for p in &o.props {
                    match p {
                        ast::ObjectPatProp::KeyValue(kv) => {
                            self.collect_pat_symbols(&kv.value, out)
                        }
                        ast::ObjectPatProp::Assign(a) => {
                            let scope = self.def_scope_of(a.key.span);
                            out.push(self.node(Ref::DefinedSymbol(a.key.sym.as_str(), scope)));
                        }
                        ast::ObjectPatProp::Rest(r) => self.collect_pat_symbols(&r.arg, out),
                    }
                }
            }
            ast::Pat::Assign(a) => self.collect_pat_symbols(&a.left, out),
            ast::Pat::Rest(r) => self.collect_pat_symbols(&r.arg, out),
            ast::Pat::Expr(_) | ast::Pat::Invalid(_) => {}
        }
    }

    /// The Babel scope uid recorded for a span (0 if not an identifier).
    fn scope_of(&self, span: Span) -> i64 {
        self.cx
            .scopes
            .get(&(span.lo.0, span.hi.0))
            .copied()
            .unwrap_or(0) as i64
    }

    /// The defScopeUid for a binding span: `var` bindings hoist to their
    /// enclosing function/program scope; others use their textual scope.
    fn def_scope_of(&self, span: Span) -> i64 {
        match self.cx.var_defs.get(&(span.lo.0, span.hi.0)) {
            Some(s) => *s as i64,
            None => self.scope_of(span),
        }
    }

    fn member_field(&self, m: &'a ast::MemberExpr, key: &str) -> Field<'a> {
        match key {
            "object" => self.expr(&m.obj),
            "computed" => Field::Bool(matches!(m.prop, ast::MemberProp::Computed(_))),
            "property" => match &m.prop {
                ast::MemberProp::Ident(n) => self.node(Ref::Name(n)),
                ast::MemberProp::Computed(c) => self.expr(&c.expr),
                ast::MemberProp::PrivateName(p) => self.node(Ref::PrivateName(p)),
            },
            _ => Field::Absent,
        }
    }

    /// `super.x` / `super[x]`: a `MemberExpression` with a `Super` object.
    fn super_prop_field(&self, m: &'a ast::SuperPropExpr, key: &str) -> Field<'a> {
        match key {
            "object" => self.node(Ref::Super(&m.obj)),
            "computed" => Field::Bool(matches!(m.prop, ast::SuperProp::Computed(_))),
            "property" => match &m.prop {
                ast::SuperProp::Ident(n) => self.node(Ref::Name(n)),
                ast::SuperProp::Computed(c) => self.expr(&c.expr),
            },
            _ => Field::Absent,
        }
    }

    fn expr_field(&self, e: &'a ast::Expr, key: &str) -> Field<'a> {
        match e {
            ast::Expr::Ident(i) if key == "name" => Field::Str(i.sym.as_str()),
            ast::Expr::Lit(l) => self.lit_field(l, key),
            ast::Expr::Paren(p) if key == "expression" => self.expr(&p.expr),
            ast::Expr::Member(m) => self.member_field(m, key),
            ast::Expr::SuperProp(m) => self.super_prop_field(m, key),
            ast::Expr::Unary(u) => match key {
                "operator" => Field::Str(unary_op(u.op)),
                "prefix" => Field::Bool(true),
                "argument" => self.expr(&u.arg),
                _ => Field::Absent,
            },
            ast::Expr::Update(u) => match key {
                "operator" => Field::Str(update_op(u.op)),
                "prefix" => Field::Bool(u.prefix),
                "argument" => self.expr(&u.arg),
                _ => Field::Absent,
            },
            ast::Expr::Bin(b) => match key {
                "operator" => Field::Str(bin_op(b.op)),
                "left" => self.expr(&b.left),
                "right" => self.expr(&b.right),
                _ => Field::Absent,
            },
            ast::Expr::Assign(a) => match key {
                "operator" => Field::Str(assign_op(a.op)),
                "left" => self.assign_target(&a.left),
                "right" => self.expr(&a.right),
                _ => Field::Absent,
            },
            ast::Expr::Cond(c) => match key {
                "test" => self.expr(&c.test),
                "consequent" => self.expr(&c.cons),
                "alternate" => self.expr(&c.alt),
                _ => Field::Absent,
            },
            ast::Expr::Seq(s) if key == "expressions" => {
                Field::List(s.exprs.iter().map(|x| self.expr(x)).collect())
            }
            ast::Expr::Call(c) => match key {
                "callee" => match &c.callee {
                    ast::Callee::Expr(e) => self.expr(e),
                    ast::Callee::Import(i) => self.node(Ref::Import(i)),
                    ast::Callee::Super(s) => self.node(Ref::Super(s)),
                },
                "arguments" => Field::List(c.args.iter().map(|a| self.arg(a)).collect()),
                _ => Field::Absent,
            },
            ast::Expr::New(n) => match key {
                "callee" => self.expr(&n.callee),
                "arguments" => match &n.args {
                    Some(args) => Field::List(args.iter().map(|a| self.arg(a)).collect()),
                    None => Field::List(vec![]),
                },
                _ => Field::Absent,
            },
            ast::Expr::Array(a) if key == "elements" => Field::List(
                a.elems
                    .iter()
                    .map(|el| match el {
                        Some(eos) => self.arg(eos),
                        None => Field::Null,
                    })
                    .collect(),
            ),
            ast::Expr::Tpl(t) => self.tpl_field(t, key),
            ast::Expr::TaggedTpl(t) => match key {
                "tag" => self.expr(&t.tag),
                "quasi" => self.node(Ref::TplLit(&t.tpl)),
                _ => Field::Absent,
            },
            ast::Expr::Arrow(a) => match key {
                "id" => Field::Null,
                "params" => Field::List(a.params.iter().map(|p| self.pat(p)).collect()),
                "body" => match &*a.body {
                    ast::BlockStmtOrExpr::BlockStmt(b) => self.node(Ref::FnBody(b)),
                    ast::BlockStmtOrExpr::Expr(e) => self.expr(e),
                },
                "generator" => Field::Bool(a.is_generator),
                "async" => Field::Bool(a.is_async),
                _ => Field::Absent,
            },
            ast::Expr::Fn(f) => self.function_field(f.ident.as_ref(), &f.function, key),
            ast::Expr::Yield(y) => match key {
                "argument" => match &y.arg {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "delegate" => Field::Bool(y.delegate),
                _ => Field::Absent,
            },
            ast::Expr::Await(a) if key == "argument" => self.expr(&a.arg),
            // An optional chain: each access maps to an OptionalMemberExpression
            // whose object recursively maps (a nested chain stays optional, a
            // plain member stays MemberExpression). The base is a `MemberExpr`,
            // so object/computed/property reuse `member_field`.
            ast::Expr::OptChain(o) => match &*o.base {
                ast::OptChainBase::Member(m) => {
                    if key == "optional" {
                        Field::Bool(o.optional)
                    } else {
                        self.member_field(m, key)
                    }
                }
                ast::OptChainBase::Call(_) => Field::Absent,
            },
            ast::Expr::MetaProp(m) => self.meta_prop_field(m, key),
            ast::Expr::Object(o) if key == "properties" => Field::List(
                o.props
                    .iter()
                    .map(|p| match p {
                        ast::PropOrSpread::Spread(s) => self.node(Ref::Spread(&s.expr)),
                        ast::PropOrSpread::Prop(prop) => self.node(Ref::ObjProp(prop)),
                    })
                    .collect(),
            ),
            ast::Expr::Class(c) => self.class_field(c.ident.as_ref(), &c.class, key),
            _ => Field::Absent,
        }
    }

    /// Shared `ClassDeclaration`/`ClassExpression` fields.
    fn class_field(
        &self,
        ident: Option<&'a ast::Ident>,
        class: &'a ast::Class,
        key: &str,
    ) -> Field<'a> {
        match key {
            "id" => match ident {
                Some(i) => self.node(Ref::Ident(i)),
                None => Field::Null,
            },
            "superClass" => match &class.super_class {
                Some(e) => self.expr(e),
                None => Field::Null,
            },
            "body" => self.node(Ref::ClassBody(class)),
            _ => Field::Absent,
        }
    }

    fn class_member_field(&self, m: &'a ast::ClassMember, key: &str) -> Field<'a> {
        match m {
            ast::ClassMember::ClassProp(p) => match key {
                "key" => self.prop_name(&p.key),
                "computed" => Field::Bool(matches!(p.key, ast::PropName::Computed(_))),
                "value" => match &p.value {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "static" => Field::Bool(p.is_static),
                _ => Field::Absent,
            },
            ast::ClassMember::PrivateProp(p) => match key {
                "key" => self.node(Ref::PrivateName(&p.key)),
                "value" => match &p.value {
                    Some(e) => self.expr(e),
                    None => Field::Null,
                },
                "static" => Field::Bool(p.is_static),
                _ => Field::Absent,
            },
            ast::ClassMember::Method(m) => match key {
                "key" => self.prop_name(&m.key),
                "computed" => Field::Bool(matches!(m.key, ast::PropName::Computed(_))),
                "static" => Field::Bool(m.is_static),
                "kind" => Field::Str(method_kind(m.kind)),
                _ => self.method_field(&m.key, &m.function, method_kind(m.kind), key),
            },
            ast::ClassMember::PrivateMethod(m) => match key {
                "key" => self.node(Ref::PrivateName(&m.key)),
                "static" => Field::Bool(m.is_static),
                "kind" => Field::Str(method_kind(m.kind)),
                "params" => {
                    Field::List(m.function.params.iter().map(|p| self.pat(&p.pat)).collect())
                }
                "body" => match &m.function.body {
                    Some(b) => self.node(Ref::FnBody(b)),
                    None => Field::Null,
                },
                "generator" => Field::Bool(m.function.is_generator),
                "async" => Field::Bool(m.function.is_async),
                _ => Field::Absent,
            },
            // Babel models a constructor as a `ClassMethod` with kind
            // `constructor`.
            ast::ClassMember::Constructor(c) => match key {
                "key" => self.prop_name(&c.key),
                "computed" => Field::Bool(matches!(c.key, ast::PropName::Computed(_))),
                "static" => Field::Bool(false),
                "kind" => Field::Str("constructor"),
                "params" => Field::List(
                    c.params
                        .iter()
                        .filter_map(|p| match p {
                            ast::ParamOrTsParamProp::Param(param) => Some(self.pat(&param.pat)),
                            ast::ParamOrTsParamProp::TsParamProp(_) => None,
                        })
                        .collect(),
                ),
                "body" => match &c.body {
                    Some(b) => self.node(Ref::FnBody(b)),
                    None => Field::Null,
                },
                "generator" | "async" => Field::Bool(false),
                _ => Field::Absent,
            },
            _ => Field::Absent,
        }
    }

    /// A property key (`PropName`) presented as the matching Babel key node.
    fn prop_name(&self, n: &'a ast::PropName) -> Field<'a> {
        match n {
            ast::PropName::Ident(i) => self.node(Ref::Name(i)),
            ast::PropName::Str(s) => self.node(Ref::StrLit(s)),
            ast::PropName::Num(n) => self.node(Ref::NumLit(n)),
            ast::PropName::Computed(c) => self.expr(&c.expr),
            ast::PropName::BigInt(_) => Field::Absent,
        }
    }

    fn object_pat_field(&self, o: &'a ast::ObjectPat, key: &str) -> Field<'a> {
        match key {
            "properties" => Field::List(
                o.props
                    .iter()
                    .map(|p| self.node(Ref::ObjPatProp(p)))
                    .collect(),
            ),
            _ => Field::Absent,
        }
    }

    fn obj_pat_prop_field(&self, p: &'a ast::ObjectPatProp, key: &str) -> Field<'a> {
        match p {
            // `{ k: pat }` -> ObjectProperty with a pattern value.
            ast::ObjectPatProp::KeyValue(kv) => match key {
                "key" => self.prop_name(&kv.key),
                "value" => self.pat(&kv.value),
                "computed" => Field::Bool(matches!(kv.key, ast::PropName::Computed(_))),
                "shorthand" => Field::Bool(false),
                _ => Field::Absent,
            },
            // `{ k }` -> shorthand ObjectProperty (key and value are the binding).
            ast::ObjectPatProp::Assign(a) if a.value.is_none() => match key {
                "key" | "value" => self.node(Ref::BindingIdent(&a.key)),
                "computed" => Field::Bool(false),
                "shorthand" => Field::Bool(true),
                _ => Field::Absent,
            },
            // `{ k = default }` -> shorthand ObjectProperty whose value is an
            // AssignmentPattern (key as left, default as right).
            ast::ObjectPatProp::Assign(a) => match key {
                "key" => self.node(Ref::BindingIdent(&a.key)),
                "value" => self.node(Ref::AssignPatPropDefault(a)),
                "computed" => Field::Bool(false),
                "shorthand" => Field::Bool(true),
                _ => Field::Absent,
            },
            ast::ObjectPatProp::Rest(r) if key == "argument" => self.pat(&r.arg),
            _ => Field::Absent,
        }
    }

    fn obj_prop_field(&self, p: &'a ast::Prop, key: &str) -> Field<'a> {
        match p {
            ast::Prop::Shorthand(i) => match key {
                "key" | "value" => self.node(Ref::Ident(i)),
                "computed" => Field::Bool(false),
                "shorthand" => Field::Bool(true),
                _ => Field::Absent,
            },
            ast::Prop::KeyValue(kv) => match key {
                "key" => self.prop_name(&kv.key),
                "value" => self.expr(&kv.value),
                "computed" => Field::Bool(matches!(kv.key, ast::PropName::Computed(_))),
                "shorthand" => Field::Bool(false),
                _ => Field::Absent,
            },
            ast::Prop::Method(m) => self.method_field(&m.key, &m.function, "method", key),
            ast::Prop::Getter(g) => match key {
                "key" => self.prop_name(&g.key),
                "computed" => Field::Bool(matches!(g.key, ast::PropName::Computed(_))),
                "params" => Field::List(vec![]),
                "body" => match &g.body {
                    Some(b) => self.node(Ref::FnBody(b)),
                    None => Field::Null,
                },
                "generator" | "async" => Field::Bool(false),
                "kind" => Field::Str("get"),
                _ => Field::Absent,
            },
            ast::Prop::Setter(s) => match key {
                "key" => self.prop_name(&s.key),
                "computed" => Field::Bool(matches!(s.key, ast::PropName::Computed(_))),
                "params" => Field::List(vec![self.pat(&s.param)]),
                "body" => match &s.body {
                    Some(b) => self.node(Ref::FnBody(b)),
                    None => Field::Null,
                },
                "generator" | "async" => Field::Bool(false),
                "kind" => Field::Str("set"),
                _ => Field::Absent,
            },
            ast::Prop::Assign(_) => Field::Absent,
        }
    }

    /// Shared `ObjectMethod` fields (also reused for regular methods).
    fn method_field(
        &self,
        prop_key: &'a ast::PropName,
        func: &'a ast::Function,
        kind: &'static str,
        key: &str,
    ) -> Field<'a> {
        match key {
            "key" => self.prop_name(prop_key),
            "computed" => Field::Bool(matches!(prop_key, ast::PropName::Computed(_))),
            "params" => Field::List(func.params.iter().map(|p| self.pat(&p.pat)).collect()),
            "body" => match &func.body {
                Some(b) => self.node(Ref::FnBody(b)),
                None => Field::Null,
            },
            "generator" => Field::Bool(func.is_generator),
            "async" => Field::Bool(func.is_async),
            "kind" => Field::Str(kind),
            _ => Field::Absent,
        }
    }

    /// Shared `FunctionDeclaration`/`FunctionExpression` fields.
    fn function_field(
        &self,
        ident: Option<&'a ast::Ident>,
        func: &'a ast::Function,
        key: &str,
    ) -> Field<'a> {
        match key {
            "id" => match ident {
                Some(i) => self.node(Ref::Ident(i)),
                None => Field::Null,
            },
            "generator" => Field::Bool(func.is_generator),
            "async" => Field::Bool(func.is_async),
            "params" => Field::List(func.params.iter().map(|p| self.pat(&p.pat)).collect()),
            "body" => match &func.body {
                Some(b) => self.node(Ref::FnBody(b)),
                None => Field::Null,
            },
            _ => Field::Absent,
        }
    }

    fn module_decl_field(&self, m: &'a ast::ModuleDecl, key: &str) -> Field<'a> {
        match m {
            ast::ModuleDecl::Import(d) => match key {
                "source" => self.node(Ref::StrLit(&d.src)),
                "specifiers" => Field::List(
                    d.specifiers
                        .iter()
                        .map(|s| self.node(Ref::ImportSpec(s)))
                        .collect(),
                ),
                _ => Field::Absent,
            },
            ast::ModuleDecl::ExportNamed(d) => match key {
                "specifiers" => Field::List(
                    d.specifiers
                        .iter()
                        .map(|s| self.node(Ref::ExportSpec(s)))
                        .collect(),
                ),
                "source" => match &d.src {
                    Some(s) => self.node(Ref::StrLit(s)),
                    None => Field::Null,
                },
                "declaration" => Field::Null,
                _ => Field::Absent,
            },
            ast::ModuleDecl::ExportDecl(d) => match key {
                "declaration" => self.node(Ref::Decl(&d.decl)),
                "specifiers" => Field::List(vec![]),
                "source" => Field::Null,
                _ => Field::Absent,
            },
            ast::ModuleDecl::ExportDefaultDecl(d) if key == "declaration" => {
                self.node(Ref::DefaultDecl(&d.decl))
            }
            ast::ModuleDecl::ExportDefaultExpr(d) if key == "declaration" => self.expr(&d.expr),
            ast::ModuleDecl::ExportAll(d) if key == "source" => self.node(Ref::StrLit(&d.src)),
            _ => Field::Absent,
        }
    }

    fn import_spec_field(&self, s: &'a ast::ImportSpecifier, key: &str) -> Field<'a> {
        let local = match s {
            ast::ImportSpecifier::Named(n) => &n.local,
            ast::ImportSpecifier::Default(d) => &d.local,
            ast::ImportSpecifier::Namespace(n) => &n.local,
        };
        match key {
            "local" => self.node(Ref::Ident(local)),
            "imported" => match s {
                ast::ImportSpecifier::Named(n) => match &n.imported {
                    Some(name) => self.module_export_name(name),
                    None => self.node(Ref::Ident(local)), // shorthand: imported == local
                },
                _ => Field::Absent,
            },
            "definedSymbols" => {
                let scope = self.scope_of(local.span);
                Field::List(vec![
                    self.node(Ref::DefinedSymbol(local.sym.as_str(), scope))
                ])
            }
            _ => Field::Absent,
        }
    }

    fn export_spec_field(&self, s: &'a ast::ExportSpecifier, key: &str) -> Field<'a> {
        match s {
            ast::ExportSpecifier::Named(n) => match key {
                "local" => self.module_export_name(&n.orig),
                "exported" => match &n.exported {
                    Some(e) => self.module_export_name(e),
                    None => self.module_export_name(&n.orig),
                },
                _ => Field::Absent,
            },
            _ => Field::Absent,
        }
    }

    fn module_export_name(&self, n: &'a ast::ModuleExportName) -> Field<'a> {
        match n {
            ast::ModuleExportName::Ident(i) => self.node(Ref::Ident(i)),
            ast::ModuleExportName::Str(s) => self.node(Ref::StrLit(s)),
        }
    }

    fn meta_prop_field(&self, m: &'a ast::MetaPropExpr, key: &str) -> Field<'a> {
        let (meta, prop) = match m.kind {
            ast::MetaPropKind::ImportMeta => ("import", "meta"),
            ast::MetaPropKind::NewTarget => ("new", "target"),
        };
        let lo = m.span.lo.0;
        let meta_span = Span {
            lo: BytePos(lo),
            hi: BytePos(lo + meta.len() as u32),
        };
        let prop_span = Span {
            lo: BytePos(lo + meta.len() as u32 + 1),
            hi: m.span.hi,
        };
        match key {
            "meta" => self.node(Ref::SynthIdent(meta_span, meta)),
            "property" => self.node(Ref::SynthIdent(prop_span, prop)),
            _ => Field::Absent,
        }
    }

    fn tpl_field(&self, t: &'a ast::Tpl, key: &str) -> Field<'a> {
        match key {
            "quasis" => Field::List(
                t.quasis
                    .iter()
                    .map(|q| self.node(Ref::TplElement(q)))
                    .collect(),
            ),
            "expressions" => Field::List(t.exprs.iter().map(|e| self.expr(e)).collect()),
            _ => Field::Absent,
        }
    }

    /// A bigint's `value`/`rawValue`: the raw literal minus the trailing `n` and
    /// any numeric separators (`0x33333333`, or `1000000` for `1_000_000n`),
    /// keeping the `0x`/`0b`/`0o` prefix. Falls back to the decimal value if swc
    /// kept no raw.
    fn bigint_value(&self, n: &'a ast::BigInt) -> Field<'a> {
        match n.raw.as_deref() {
            Some(r) => {
                let trimmed = r.strip_suffix('n').unwrap_or(r);
                if trimmed.contains('_') {
                    Field::Str(self.bump.alloc_str(&trimmed.replace('_', "")))
                } else {
                    Field::Str(trimmed)
                }
            }
            None => Field::Str(self.bump.alloc_str(&n.value.to_string())),
        }
    }

    fn lit_field(&self, l: &'a ast::Lit, key: &str) -> Field<'a> {
        match (l, key) {
            (ast::Lit::Num(n), "value") => Field::Float(n.value),
            (ast::Lit::Num(n), "extra") => self.node(Ref::NumExtra(n)),
            (ast::Lit::Str(s), "value") => str_value(s),
            (ast::Lit::Str(s), "extra") => self.node(Ref::StrExtra(s)),
            (ast::Lit::Bool(b), "value") => Field::Bool(b.value),
            (ast::Lit::Regex(r), "pattern") => Field::Str(&r.exp),
            (ast::Lit::Regex(r), "flags") => Field::Str(&r.flags),
            (ast::Lit::Regex(r), "extra") => self.node(Ref::RegexExtra(r)),
            // JSIR `value`/`rawValue` are the raw literal text minus the `n`
            // suffix (keeping any `0x`/`0b`/`0o` prefix), not the decimal value.
            (ast::Lit::BigInt(n), "value") => self.bigint_value(n),
            (ast::Lit::BigInt(n), "extra") => self.node(Ref::BigIntExtra(n)),
            _ => Field::Absent,
        }
    }

    fn num_extra_field(&self, n: &'a ast::Number, key: &str) -> Field<'a> {
        match key {
            "raw" => match n.raw.as_deref() {
                Some(r) => Field::Str(r),
                // swc dropped the source text (synthesized node): re-render it.
                None => Field::Str(self.bump.alloc_str(&render_number(n.value))),
            },
            "rawValue" => Field::Float(n.value),
            _ => Field::Absent,
        }
    }
}

fn tpl_value_field<'a>(t: &'a ast::TplElement, key: &str) -> Field<'a> {
    match key {
        "cooked" => match t.cooked.as_ref().and_then(|c| c.as_str()) {
            Some(c) => Field::Str(c),
            None => Field::Absent,
        },
        "raw" => Field::Str(t.raw.as_str()),
        _ => Field::Absent,
    }
}

/// A string literal's decoded `value` (WTF-8 atom; `None` only for the rare
/// lone-surrogate case).
fn str_value(s: &ast::Str) -> Field<'_> {
    match s.value.as_str() {
        Some(v) => Field::Str(v),
        None => Field::Absent,
    }
}

/// The leading directive prologue of a statement list: consecutive
/// string-literal expression statements (Babel's `Program.directives`), plus the
/// remaining statements.
fn split_directives(body: &[ast::Stmt]) -> (Vec<&ast::Str>, &[ast::Stmt]) {
    let mut dirs = Vec::new();
    let mut i = 0;
    while let Some(ast::Stmt::Expr(es)) = body.get(i) {
        if let ast::Expr::Lit(ast::Lit::Str(s)) = &*es.expr {
            dirs.push(s);
            i += 1;
        } else {
            break;
        }
    }
    (dirs, &body[i..])
}

fn str_extra_field<'a>(s: &'a ast::Str, key: &str) -> Field<'a> {
    match key {
        "raw" => match s.raw.as_deref() {
            Some(r) => Field::Str(r),
            None => Field::Absent,
        },
        "rawValue" => match s.value.as_str() {
            Some(v) => Field::Str(v),
            None => Field::Absent,
        },
        _ => Field::Absent,
    }
}

/// Render a number the way Babel's `raw` would for a synthesized literal.
fn render_number(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn stmt_type(s: &ast::Stmt) -> &'static str {
    match s {
        ast::Stmt::Expr(_) => "ExpressionStatement",
        ast::Stmt::Empty(_) => "EmptyStatement",
        ast::Stmt::Debugger(_) => "DebuggerStatement",
        ast::Stmt::Block(_) => "BlockStatement",
        ast::Stmt::Return(_) => "ReturnStatement",
        ast::Stmt::Throw(_) => "ThrowStatement",
        ast::Stmt::With(_) => "WithStatement",
        ast::Stmt::If(_) => "IfStatement",
        ast::Stmt::While(_) => "WhileStatement",
        ast::Stmt::DoWhile(_) => "DoWhileStatement",
        ast::Stmt::Break(_) => "BreakStatement",
        ast::Stmt::Continue(_) => "ContinueStatement",
        ast::Stmt::For(_) => "ForStatement",
        ast::Stmt::ForIn(_) => "ForInStatement",
        ast::Stmt::ForOf(_) => "ForOfStatement",
        ast::Stmt::Try(_) => "TryStatement",
        ast::Stmt::Switch(_) => "SwitchStatement",
        ast::Stmt::Labeled(_) => "LabeledStatement",
        ast::Stmt::Decl(ast::Decl::Var(_)) => "VariableDeclaration",
        ast::Stmt::Decl(ast::Decl::Fn(_)) => "FunctionDeclaration",
        ast::Stmt::Decl(ast::Decl::Class(_)) => "ClassDeclaration",
        _ => "UnsupportedStatement",
    }
}

fn expr_type(e: &ast::Expr) -> &'static str {
    match e {
        ast::Expr::Ident(_) => "Identifier",
        ast::Expr::Lit(l) => lit_type(l),
        ast::Expr::Unary(_) => "UnaryExpression",
        ast::Expr::Update(_) => "UpdateExpression",
        ast::Expr::Bin(b) => {
            if is_logical(b.op) {
                "LogicalExpression"
            } else {
                "BinaryExpression"
            }
        }
        ast::Expr::Assign(_) => "AssignmentExpression",
        ast::Expr::Cond(_) => "ConditionalExpression",
        ast::Expr::Seq(_) => "SequenceExpression",
        ast::Expr::Call(_) => "CallExpression",
        ast::Expr::New(_) => "NewExpression",
        ast::Expr::Array(_) => "ArrayExpression",
        ast::Expr::Member(_) => "MemberExpression",
        ast::Expr::SuperProp(_) => "MemberExpression",
        ast::Expr::OptChain(o) => match &*o.base {
            ast::OptChainBase::Member(_) => "OptionalMemberExpression",
            ast::OptChainBase::Call(_) => "OptionalCallExpression",
        },
        ast::Expr::Paren(_) => "ParenthesizedExpression",
        ast::Expr::Tpl(_) => "TemplateLiteral",
        ast::Expr::TaggedTpl(_) => "TaggedTemplateExpression",
        ast::Expr::Arrow(_) => "ArrowFunctionExpression",
        ast::Expr::Fn(_) => "FunctionExpression",
        ast::Expr::Yield(_) => "YieldExpression",
        ast::Expr::Await(_) => "AwaitExpression",
        ast::Expr::MetaProp(_) => "MetaProperty",
        ast::Expr::Object(_) => "ObjectExpression",
        ast::Expr::Class(_) => "ClassExpression",
        ast::Expr::This(_) => "ThisExpression",
        _ => "UnsupportedExpression",
    }
}

fn pat_type(p: &ast::Pat) -> &'static str {
    match p {
        ast::Pat::Ident(_) => "Identifier",
        ast::Pat::Array(_) => "ArrayPattern",
        ast::Pat::Object(_) => "ObjectPattern",
        ast::Pat::Assign(_) => "AssignmentPattern",
        ast::Pat::Rest(_) => "RestElement",
        _ => "UnsupportedPattern",
    }
}

fn lit_type(l: &ast::Lit) -> &'static str {
    match l {
        ast::Lit::Str(_) => "StringLiteral",
        ast::Lit::Bool(_) => "BooleanLiteral",
        ast::Lit::Null(_) => "NullLiteral",
        ast::Lit::Num(_) => "NumericLiteral",
        ast::Lit::BigInt(_) => "BigIntLiteral",
        ast::Lit::Regex(_) => "RegExpLiteral",
        ast::Lit::JSXText(_) => "UnsupportedLiteral",
    }
}

fn unary_op(op: ast::UnaryOp) -> &'static str {
    match op {
        ast::UnaryOp::Minus => "-",
        ast::UnaryOp::Plus => "+",
        ast::UnaryOp::Bang => "!",
        ast::UnaryOp::Tilde => "~",
        ast::UnaryOp::TypeOf => "typeof",
        ast::UnaryOp::Void => "void",
        ast::UnaryOp::Delete => "delete",
    }
}

fn update_op(op: ast::UpdateOp) -> &'static str {
    match op {
        ast::UpdateOp::PlusPlus => "++",
        ast::UpdateOp::MinusMinus => "--",
    }
}

fn assign_op(op: ast::AssignOp) -> &'static str {
    use ast::AssignOp::*;
    match op {
        Assign => "=",
        AddAssign => "+=",
        SubAssign => "-=",
        MulAssign => "*=",
        DivAssign => "/=",
        ModAssign => "%=",
        LShiftAssign => "<<=",
        RShiftAssign => ">>=",
        ZeroFillRShiftAssign => ">>>=",
        BitOrAssign => "|=",
        BitXorAssign => "^=",
        BitAndAssign => "&=",
        ExpAssign => "**=",
        AndAssign => "&&=",
        OrAssign => "||=",
        NullishAssign => "??=",
    }
}

fn is_logical(op: ast::BinaryOp) -> bool {
    matches!(
        op,
        ast::BinaryOp::LogicalAnd | ast::BinaryOp::LogicalOr | ast::BinaryOp::NullishCoalescing
    )
}

fn bin_op(op: ast::BinaryOp) -> &'static str {
    use ast::BinaryOp::*;
    match op {
        EqEq => "==",
        NotEq => "!=",
        EqEqEq => "===",
        NotEqEq => "!==",
        Lt => "<",
        LtEq => "<=",
        Gt => ">",
        GtEq => ">=",
        LShift => "<<",
        RShift => ">>",
        ZeroFillRShift => ">>>",
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Mod => "%",
        BitOr => "|",
        BitXor => "^",
        BitAnd => "&",
        In => "in",
        InstanceOf => "instanceof",
        Exp => "**",
        LogicalOr => "||",
        LogicalAnd => "&&",
        NullishCoalescing => "??",
    }
}

/// Parse JS source and lower it straight to JSIR IR via the `AstNode` trait,
/// with no intermediate AST tree.
pub fn source_to_ir(src: &str) -> Result<jsir_ir::Op, String> {
    let cm = SourceMap::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Anon), src.to_string());
    let file_start = fm.start_pos;
    let comments = SingleThreadedComments::default();
    let lexer = Lexer::new(
        Syntax::Es(swc_ecma_parser::EsSyntax { jsx: true, ..Default::default() }),
        EsVersion::EsNext,
        StringInput::from(&*fm),
        Some(&comments),
    );
    let mut parser = Parser::new_from(lexer);
    let mut program = parser
        .parse_program()
        .map_err(|e| format!("swc parse error: {e:?}"))?;
    drop(parser); // release the borrow of `fm` so `cm` can move into `Cx`

    // Desugar JSX to React.createElement calls before scope/IR construction.
    crate::jsx::desugar(&mut program);

    let mut scopes = ScopeAssigner::default();
    program.visit_with(&mut scopes);

    // Flatten swc's leading/trailing comment maps into one source-ordered list
    // (Babel's `File.comments`); a comment can be filed under both maps, so dedup.
    let (leading, trailing) = comments.take_all();
    let mut all: Vec<Comment> = Vec::new();
    for map in [leading, trailing] {
        for v in map.borrow().values() {
            all.extend(v.iter().cloned());
        }
    }
    all.sort_by_key(|c| c.span.lo.0);
    all.dedup_by_key(|c| (c.span.lo.0, c.span.hi.0));

    let cx = Cx {
        file_start,
        line_starts: compute_line_starts(src),
        src: src.to_string(),
        scopes: scopes.map,
        var_defs: scopes.var_defs,
        parent: scopes.parent,
        decls: scopes.decls,
        comments: all,
    };
    let bump = Bump::new();
    let file = W {
        node: Ref::File(&program),
        bump: &bump,
        cx: &cx,
    };
    jsir_convert::ast2hir(&file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The end goal: real JS, parsed by swc, lowered straight to IR with no
    /// intermediate tree, byte-comparable (FileCheck-equivalent) to upstream's
    /// golden `jshir.mlir`. We report coverage across the whole corpus and
    /// assert the node types implemented so far all match.
    #[test]
    fn swc_source_to_ir_matches_corpus() {
        let mut matched = Vec::new();
        let mut diverged = Vec::new();
        let mut errored = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(expected) = f.expected_jshir() else {
                continue;
            };
            let src = f.input_js().expect("input.js");
            match source_to_ir(&src) {
                Ok(op) => {
                    let actual = op.print();
                    if jsir_oracle::filecheck_equivalent(&expected, &actual) {
                        matched.push(f.name.clone());
                    } else {
                        diverged.push(f.name.clone());
                    }
                }
                Err(e) => errored.push(format!("{}: {e}", f.name)),
            }
        }
        eprintln!(
            "swc->IR: {} matched, {} diverged, {} errored (of {} fixtures)",
            matched.len(),
            diverged.len(),
            errored.len(),
            matched.len() + diverged.len() + errored.len()
        );
        eprintln!("matched: {}", matched.join(", "));
        eprintln!("diverged: {}", diverged.join(", "));

        // The whole corpus must lower swc -> IR byte-equivalent (FileCheck),
        // with nothing diverging or erroring.
        assert!(
            diverged.is_empty() && errored.is_empty(),
            "swc->IR regressed:\ndiverged={diverged:?}\nerrored={errored:?}"
        );
        assert_eq!(matched.len(), 46, "expected all 46 fixtures to match");
    }

    /// Strip the formatting-dependent trivia from printed IR so two IRs from
    /// differently-formatted-but-equivalent source compare equal: source
    /// positions (`<L n C n>`), the integer offsets/scope inside `#jsir<...>`
    /// attributes, and comments (which the code generator drops). What remains
    /// is the semantic structure: op names, operands, regions, names, operators.
    fn normalize_ir(ir: &str) -> String {
        use regex::Regex;
        let comments = Regex::new(r"comments = \[[^\]]*\]").unwrap();
        let loc = Regex::new(r"<L \d+ C \d+>").unwrap();
        let ints = Regex::new(r", \d+").unwrap();
        let s = comments.replace_all(ir, "comments = []");
        let s = loc.replace_all(&s, "<L>");
        ints.replace_all(&s, ", N").into_owned()
    }

    /// Full end-to-end loop: `js -> swc -> ir -> swc -> js -> swc -> ir`. The two
    /// IRs must match modulo formatting trivia (positions/offsets/comments),
    /// proving the round trip preserves the program's *structure*: lifting the IR
    /// back to JS and re-lowering reproduces the same IR. Scales to any JS source
    /// (the basis for the test262 self-consistency oracle).
    #[test]
    fn full_round_trip_is_stable() {
        let mut stable = 0;
        let mut failures = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            if f.expected_jshir().is_none() {
                continue;
            }
            let src = f.input_js().expect("input.js");
            let result = (|| {
                let ir1 = source_to_ir(&src)?;
                let js2 = crate::ir_to_source(&ir1)?;
                let ir2 = source_to_ir(&js2)?;
                Ok::<_, String>((ir1.print(), ir2.print(), js2))
            })();
            match result {
                Ok((ir1, ir2, _)) if normalize_ir(&ir1) == normalize_ir(&ir2) => stable += 1,
                Ok((ir1, ir2, js2)) => failures.push(format!(
                    "{}: structure changed across round trip\n--js2--\n{js2}\n--ir1--\n{}\n--ir2--\n{}",
                    f.name,
                    normalize_ir(&ir1),
                    normalize_ir(&ir2)
                )),
                Err(e) => failures.push(format!("{}: {e}", f.name)),
            }
        }
        eprintln!("full round trip: {stable} stable, {} not yet", failures.len());
        // 38/46 round-trip structurally stable. The remaining 8 lose structure in
        // `ir -> js` (the `to_swc` output converter drops parens/directives and
        // is incomplete for some class/object shapes); tracked, not yet gated.
        // This is a floor so regressions are caught.
        assert!(
            stable >= 38,
            "full round-trip stability regressed below 38: {stable} stable\n{}",
            failures.join("\n\n")
        );
    }
}


