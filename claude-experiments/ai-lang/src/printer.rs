//! The "view" pretty-printer: canonical `ast::Def` -> ai-lang surface source.
//!
//! This is the inverse of the resolver (`resolve.rs`). The content-addressed
//! store holds name-erased, de-Bruijn-indexed canonical ASTs; to let a human
//! (or an agent) *read* a definition we project it back to surface text that
//! re-parses and re-resolves to the SAME content hash.
//!
//! It lives strictly above the hashing line: it never touches `hash.rs`,
//! `codec.rs`, or the `ast.rs` identity model.
//!
//! ## What it recovers, and how
//!
//!   - **Top-level names.** A referenced hash (`TopRef`, `TypeRef`,
//!     `struct_ref`, `enum_ref`) is looked up in the codebase's name map. A
//!     referenced hash with NO name gets a deterministic synthetic alias
//!     (`def_<first8hex>` for terms, `T_<first8hex>` for types) so it is never
//!     silently dropped.
//!   - **Locals.** De Bruijn indices are rendered as deterministic readable
//!     names via a binder stack that mirrors the resolver's `env`: fn params,
//!     `let`, lambda params, and match-arm payload bindings all push binders in
//!     the exact order the resolver does, so `LocalVar(i)` resolves to the
//!     correct binder with no collisions.
//!   - **Operators.** `BuiltinRef` callees minted by the resolver
//!     (`core/i64.add`, `core/f64.lt`, `string_concat`, ...) are inverted back
//!     to their surface form (infix `+`, the `string_concat(...)` call, etc.),
//!     derived from the real resolver table so the printed text re-lowers to
//!     the same builtin.
//!   - **Field / constructor / match positions.** The canonical form uses
//!     positional field/variant indices; to print names we load the referenced
//!     struct/enum def from the codebase and read its declared field/variant
//!     names.
//!
//! ## The contract: NO silent wrong output
//!
//! Anything the printer cannot faithfully render is a HARD ERROR (a typed
//! [`PrinterError`]), never an empty string or a `<unknown>` placeholder.
//! Returning text that does not re-parse + re-resolve to the original hash is a
//! bug, so the constructs we cannot yet round-trip error loudly instead.
//!
//! ## Literal round-tripping (what the lexer actually allows)
//!
//! These cases are driven by `lexer.rs`, not guessed:
//!
//!   - **Strings round-trip in full.** The lexer's escape table is
//!     `\n \r \t \\ \" \0 \e` (`\e` = ESC, U+001B), and any other byte between
//!     the quotes is taken verbatim. We emit those escapes for readability and
//!     emit everything else literally, so EVERY `String` is expressible — there
//!     is no string hard-error case.
//!   - **Non-negative floats round-trip in full**, including ones that need
//!     exponent notation: the lexer's numeric grammar accepts `[eE][+-]?digits`.
//!     Integer-valued floats (e.g. `2.0`, `1e10`) are padded with `.0` so the
//!     lexer reads a Float rather than an Int.
//!
//! Genuinely inexpressible literals that STAY hard errors, with the grammar
//! reason for each:
//!
//!   - **Negative `IntLit` / negative `FloatLit`.** The lexer has no negative
//!     numeric literal — a leading `-` is the separate `Minus` operator. So
//!     `-5` / `-1.5` parse to `Unary{Neg, _}` and resolve to
//!     `Call(core/i64.neg | core/f64.neg, lit)`, a DIFFERENT canonical Expr than
//!     `IntLit(-5)` / `FloatLit(-1.5)`. There is no way to write a bare negative
//!     literal that re-resolves to the literal node, so we refuse. (Surface
//!     source never produces a negative literal node anyway; it produces the
//!     neg-Call, which the operator printer inverts correctly to `(-5)`.)
//!   - **Non-finite floats (NaN / ±Inf).** No surface literal exists for them.

use crate::ast::{Def, Expr, MatchArm, Pattern, Type};
use crate::codebase::Codebase;
use crate::hash::Hash;

use std::collections::HashMap;

// =============================================================================
// Errors
// =============================================================================

/// Errors from projecting a canonical def back to surface source. Modelled on
/// `DepIndexError` / `CodebaseError`: a small typed enum with a `Display` impl,
/// `std::error::Error`, and `From` for the wrapped codebase error.
#[derive(Debug)]
pub enum PrinterError {
    /// Failed to load or decode a referenced definition from the store.
    Codebase(crate::codebase::CodebaseError),
    /// A `LocalVar(i)` index pointed past the end of the binder stack — the
    /// canonical AST is malformed or the binder bookkeeping diverged from the
    /// resolver. Carries the offending index and the stack depth.
    UnboundLocal { index: u32, depth: usize },
    /// A `SelfRef` survived into a stored def. The resolver rewrites every
    /// `SelfRef` to a `TopRef`/`TypeRef` before storing, so seeing one here
    /// means the def was never stored (or the invariant was broken).
    UnexpectedSelfRef,
    /// A `TypeVar(i)` appeared but the enclosing def declares no (or too few)
    /// type parameters, so there is no surface name to print for it.
    UnboundTypeVar { index: u32, type_params: u32 },
    /// A `BuiltinRef` string the printer does not know how to invert back to
    /// surface syntax. Carries the builtin id.
    UnknownBuiltin(String),
    /// A referenced struct/enum hash did not resolve to a `Def::Struct` /
    /// `Def::Enum` with the expected field/variant index. Carries a
    /// description of what went wrong.
    BadTypeShape(String),
    /// A literal cannot be rendered in a way that re-lexes to the same value
    /// (e.g. a non-finite float, or a string with characters whose escaping
    /// this printer does not model). Carries a description.
    Unrenderable(String),
}

impl From<crate::codebase::CodebaseError> for PrinterError {
    fn from(e: crate::codebase::CodebaseError) -> Self {
        PrinterError::Codebase(e)
    }
}

impl core::fmt::Display for PrinterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PrinterError::Codebase(e) => write!(f, "codebase error: {}", e),
            PrinterError::UnboundLocal { index, depth } => write!(
                f,
                "local variable de Bruijn index {} out of range (binder depth {})",
                index, depth
            ),
            PrinterError::UnexpectedSelfRef => write!(
                f,
                "encountered a SelfRef in a stored def; the resolver should have \
                 rewritten all SelfRefs to TopRef/TypeRef before storing"
            ),
            PrinterError::UnboundTypeVar { index, type_params } => write!(
                f,
                "type variable index {} out of range (def declares {} type params)",
                index, type_params
            ),
            PrinterError::UnknownBuiltin(name) => {
                write!(f, "no surface form known for builtin `{}`", name)
            }
            PrinterError::BadTypeShape(s) => write!(f, "bad referenced type shape: {}", s),
            PrinterError::Unrenderable(s) => write!(f, "value cannot be rendered: {}", s),
        }
    }
}

impl std::error::Error for PrinterError {}

// =============================================================================
// Public entry point
// =============================================================================

/// Project the definition stored under `hash` back to ai-lang surface source.
///
/// The returned text is a single top-level `def` / `struct` / `enum` that,
/// parsed + resolved in a context where the same referenced names/hashes are in
/// scope, hashes back to `hash`.
pub fn print_def(cb: &Codebase, hash: Hash) -> Result<String, PrinterError> {
    let def = cb.load_def(&hash)?;
    let names = NameMap::build(cb);
    // Load the author's original local names (side-car), if present. We only
    // adopt them when their count matches this def's binder count exactly
    // (see `LocalNames::adopt`); otherwise we fall back to deterministic
    // `p0/p1/...` so the printer never renders names against the wrong
    // binders.
    let sidecar = cb.load_local_names(&hash)?;
    let local_names = LocalNames::adopt(sidecar, &def);
    let printer = Printer {
        cb,
        names: &names,
        local_names,
    };
    let self_name = printer.names.term_name(&hash);
    printer.print_def(&self_name, &def)
}

// =============================================================================
// Name recovery
// =============================================================================

/// Reverse view of the codebase's `name -> hash` map: `hash -> name`.
///
/// If multiple names alias the same hash we pick the lexicographically smallest
/// for determinism (the choice is irrelevant to the round-trip; any in-scope
/// name resolves to the same hash).
struct NameMap {
    by_hash: HashMap<Hash, String>,
}

impl NameMap {
    fn build(cb: &Codebase) -> Self {
        let mut by_hash: HashMap<Hash, String> = HashMap::new();
        for (name, h) in cb.names() {
            by_hash
                .entry(*h)
                .and_modify(|existing| {
                    if name < existing {
                        *existing = name.clone();
                    }
                })
                .or_insert_with(|| name.clone());
        }
        NameMap { by_hash }
    }

    /// Surface name for a term-level reference (a `TopRef`). Falls back to a
    /// deterministic synthetic alias `def_<first8hex>` when the hash has no
    /// name, so a reference is never dropped.
    fn term_name(&self, h: &Hash) -> String {
        match self.by_hash.get(h) {
            Some(n) => n.clone(),
            None => format!("def_{}", &h.to_hex()[..8]),
        }
    }

    /// Surface name for a type-level reference (a `TypeRef` / struct_ref /
    /// enum_ref). Falls back to a deterministic synthetic alias
    /// `T_<first8hex>` when the hash has no name.
    fn type_name(&self, h: &Hash) -> String {
        match self.by_hash.get(h) {
            Some(n) => n.clone(),
            None => format!("T_{}", &h.to_hex()[..8]),
        }
    }
}

// =============================================================================
// Printer
// =============================================================================

struct Printer<'a> {
    cb: &'a Codebase,
    names: &'a NameMap,
    /// The author's original local names, replayed against the binder stack
    /// in push order. `None` when no side-car was present or it was
    /// inconsistent with this def's binder count (then we fall back to the
    /// deterministic `p0/p1/...` generators).
    local_names: LocalNames,
}

/// Replays the author's original local names against the printer's binder
/// stack. The names are stored in binder-push order (params first, then
/// `let`/lambda/match binders in pre-order); `next` hands them out one at a
/// time as the printer pushes binders, in the exact same order they were
/// captured (see `resolve::local_names_for_module`).
struct LocalNames {
    /// `Some` only when the side-car's length matched the def's binder
    /// count exactly. The cursor advances per binder.
    names: Option<Vec<String>>,
    cursor: std::cell::Cell<usize>,
}

impl LocalNames {
    /// Decide whether to use `sidecar` for `def`. We adopt it only when its
    /// length equals the def's total binder count, so the replay can never
    /// drift onto the wrong binder. A mismatch (wrong arity, e.g. the def
    /// was edited and the side-car is stale, or a struct/enum that has no
    /// locals) is ignored cleanly — the printer falls back to `p0/p1/...`.
    fn adopt(sidecar: Option<Vec<String>>, def: &Def) -> Self {
        let names = match sidecar {
            Some(s) => {
                let expected = count_binders_in_def(def);
                if s.len() == expected {
                    Some(s)
                } else {
                    None
                }
            }
            None => None,
        };
        LocalNames {
            names,
            cursor: std::cell::Cell::new(0),
        }
    }

    /// Is the author's name list in use for this def?
    fn active(&self) -> bool {
        self.names.is_some()
    }

    /// Hand out the next author name (advancing the cursor). Only valid when
    /// `active()`. The cursor cannot run past the end because `adopt`
    /// verified the count matches the binder total.
    fn next(&self) -> String {
        let names = self
            .names
            .as_ref()
            .expect("LocalNames::next called while inactive");
        let i = self.cursor.get();
        self.cursor.set(i + 1);
        names[i].clone()
    }
}

/// Count the lexical binders a stored def introduces: fn parameters, plus
/// every lambda parameter, `let` binding, and match `Pattern::Var` in the
/// body. Used to validate the local-name side-car against the def shape.
fn count_binders_in_def(def: &Def) -> usize {
    match def {
        Def::Fn { params, body, .. } => params.len() + count_binders_in_expr(body),
        // Struct/enum have no locals.
        Def::Struct { .. } | Def::Enum { .. } => 0,
        // A state initializer is a zero-param expression; count any
        // binders it introduces (e.g. lambdas inside the init).
        Def::State { init, .. } => count_binders_in_expr(init),
    }
}

fn count_binders_in_expr(e: &Expr) -> usize {
    match e {
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => 0,
        Expr::Call(callee, args) => {
            count_binders_in_expr(callee)
                + args.iter().map(count_binders_in_expr).sum::<usize>()
        }
        Expr::Lambda { params, body } => params.len() + count_binders_in_expr(body),
        Expr::Let { value, body } => {
            // The `let` itself introduces one binder.
            count_binders_in_expr(value) + 1 + count_binders_in_expr(body)
        }
        Expr::Defer { cleanup, body } => {
            count_binders_in_expr(cleanup) + count_binders_in_expr(body)
        }
        Expr::StructNew { fields, .. } => {
            fields.iter().map(count_binders_in_expr).sum()
        }
        Expr::Field { base, .. } => count_binders_in_expr(base),
        Expr::EnumNew { payload, .. } => {
            payload.as_deref().map(count_binders_in_expr).unwrap_or(0)
        }
        Expr::Match { scrutinee, arms } => {
            count_binders_in_expr(scrutinee)
                + arms
                    .iter()
                    .map(|a| count_binders_in_pattern(&a.pattern) + count_binders_in_expr(&a.body))
                    .sum::<usize>()
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            count_binders_in_expr(cond)
                + count_binders_in_expr(then_branch)
                + count_binders_in_expr(else_branch)
        }
        Expr::Try { expr, .. } => count_binders_in_expr(expr),
    }
}

fn count_binders_in_pattern(p: &Pattern) -> usize {
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => {
            payload.as_deref().map(count_binders_in_pattern).unwrap_or(0)
        }
    }
}

/// A lexical binder name + how it must be referred to from the body. Mirrors
/// the resolver's `env`: index 0 is the innermost binder.
struct Binders {
    /// Names in *outermost-first* order: `stack[0]` is the outermost binder
    /// (de Bruijn index `len-1`), `stack[len-1]` is innermost (index 0).
    stack: Vec<String>,
}

impl Binders {
    fn new() -> Self {
        Binders { stack: Vec::new() }
    }
    fn push(&mut self, name: String) {
        self.stack.push(name);
    }
    fn pop(&mut self) {
        self.stack.pop();
    }
    /// Resolve a de Bruijn index (0 = innermost) to its binder name.
    fn name(&self, index: u32) -> Result<&str, PrinterError> {
        let depth = self.stack.len();
        if (index as usize) >= depth {
            return Err(PrinterError::UnboundLocal { index, depth });
        }
        // index 0 is the last element of the stack.
        Ok(&self.stack[depth - 1 - index as usize])
    }
}

impl<'a> Printer<'a> {
    // ---- Definitions ----

    fn print_def(&self, self_name: &str, def: &Def) -> Result<String, PrinterError> {
        match def {
            Def::Fn {
                is_local,
                type_params,
                params,
                ret,
                body,
            } => self.print_fn(self_name, *is_local, *type_params, params, ret, body),
            Def::Struct {
                type_params,
                fields,
            } => self.print_struct(self_name, *type_params, fields),
            Def::Enum {
                type_params,
                variants,
            } => self.print_enum(self_name, *type_params, variants),
            Def::State { ty, init } => self.print_state(self_name, ty, init),
        }
    }

    /// `state NAME: TYPE = INIT`. A node-resident singleton binding; the
    /// initializer is a zero-parameter expression.
    fn print_state(
        &self,
        self_name: &str,
        ty: &Type,
        init: &Expr,
    ) -> Result<String, PrinterError> {
        let mut out = String::new();
        let mut binders = Binders::new();
        out.push_str("state ");
        out.push_str(self_name);
        out.push_str(": ");
        out.push_str(&self.print_type(ty, 0, &[])?);
        out.push_str(" = ");
        out.push_str(&self.print_expr(init, &mut binders, 0, &[])?);
        Ok(out)
    }

    fn print_fn(
        &self,
        self_name: &str,
        is_local: bool,
        type_params: u32,
        params: &[Type],
        ret: &Type,
        body: &Expr,
    ) -> Result<String, PrinterError> {
        let tp_names = type_param_names(type_params);
        let mut out = String::new();
        out.push_str("def ");
        if is_local {
            out.push_str("local ");
        }
        out.push_str(self_name);
        out.push_str(&self.print_type_param_decl(&tp_names));
        out.push('(');
        // Parameter names: the author's originals when the side-car is
        // active, else deterministic `p0, p1, ...`. They become the
        // outermost binders, in order (param 0 is outermost).
        let mut binders = Binders::new();
        let param_names = if self.local_names.active() {
            (0..params.len()).map(|_| self.local_names.next()).collect()
        } else {
            param_names(params.len())
        };
        for (i, (pname, pty)) in param_names.iter().zip(params.iter()).enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(pname);
            out.push_str(": ");
            out.push_str(&self.print_type(pty, type_params, &tp_names)?);
        }
        // Push params as binders: param 0 is the OUTERMOST binder, matching the
        // resolver which builds `env` from params in order then counts de
        // Bruijn outward.
        for pname in &param_names {
            binders.push(pname.clone());
        }
        out.push(')');
        out.push_str(" -> ");
        out.push_str(&self.print_type(ret, type_params, &tp_names)?);
        out.push_str(" = ");
        out.push_str(&self.print_expr(body, &mut binders, type_params, &tp_names)?);
        Ok(out)
    }

    fn print_struct(
        &self,
        self_name: &str,
        type_params: u32,
        fields: &[(String, Type)],
    ) -> Result<String, PrinterError> {
        let tp_names = type_param_names(type_params);
        let mut out = String::new();
        out.push_str("struct ");
        out.push_str(self_name);
        out.push_str(&self.print_type_param_decl(&tp_names));
        out.push_str(" {");
        for (i, (fname, fty)) in fields.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push(' ');
            out.push_str(fname);
            out.push_str(": ");
            out.push_str(&self.print_type(fty, type_params, &tp_names)?);
        }
        if !fields.is_empty() {
            out.push(' ');
        }
        out.push('}');
        Ok(out)
    }

    fn print_enum(
        &self,
        self_name: &str,
        type_params: u32,
        variants: &[(String, Option<Type>)],
    ) -> Result<String, PrinterError> {
        let tp_names = type_param_names(type_params);
        let mut out = String::new();
        out.push_str("enum ");
        out.push_str(self_name);
        out.push_str(&self.print_type_param_decl(&tp_names));
        out.push_str(" {");
        for (i, (vname, payload)) in variants.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push(' ');
            out.push_str(vname);
            if let Some(pty) = payload {
                out.push('(');
                out.push_str(&self.print_type(pty, type_params, &tp_names)?);
                out.push(')');
            }
        }
        if !variants.is_empty() {
            out.push(' ');
        }
        out.push('}');
        Ok(out)
    }

    fn print_type_param_decl(&self, tp_names: &[String]) -> String {
        if tp_names.is_empty() {
            return String::new();
        }
        format!("<{}>", tp_names.join(", "))
    }

    // ---- Types ----

    fn print_type(
        &self,
        ty: &Type,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        match ty {
            Type::Builtin(name) => Ok(name.clone()),
            Type::TypeRef(h) => Ok(self.names.type_name(h)),
            Type::TypeVar(i) => {
                if (*i as usize) >= tp_names.len() {
                    return Err(PrinterError::UnboundTypeVar {
                        index: *i,
                        type_params,
                    });
                }
                Ok(tp_names[*i as usize].clone())
            }
            Type::Apply(head, args) => {
                // Head must be a name (Builtin like `Array`, or a TypeRef).
                let head_name = match head.as_ref() {
                    Type::Builtin(name) => name.clone(),
                    Type::TypeRef(h) => self.names.type_name(h),
                    other => {
                        return Err(PrinterError::BadTypeShape(format!(
                            "Apply head is not a named type: {:?}",
                            other
                        )));
                    }
                };
                let mut parts = Vec::with_capacity(args.len());
                for a in args {
                    parts.push(self.print_type(a, type_params, tp_names)?);
                }
                Ok(format!("{}<{}>", head_name, parts.join(", ")))
            }
            Type::FnType { params, ret } => {
                let mut parts = Vec::with_capacity(params.len());
                for p in params {
                    parts.push(self.print_type(p, type_params, tp_names)?);
                }
                let ret_s = self.print_type(ret, type_params, tp_names)?;
                Ok(format!("fn({}) -> {}", parts.join(", "), ret_s))
            }
            Type::SelfRef(_) => Err(PrinterError::UnexpectedSelfRef),
        }
    }

    // ---- Expressions ----

    fn print_expr(
        &self,
        e: &Expr,
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        match e {
            Expr::IntLit(v) => render_int(*v),
            Expr::FloatLit(v) => render_float(*v),
            Expr::BoolLit(b) => Ok(if *b { "true" } else { "false" }.to_owned()),
            Expr::StringLit(s) => render_string(s),

            Expr::LocalVar(i) => Ok(binders.name(*i)?.to_owned()),

            Expr::TopRef(h) => Ok(self.names.term_name(h)),

            // A node `state` reference prints as the binding's name.
            Expr::StateRef(h) => Ok(self.names.term_name(h)),

            Expr::SelfRef(_) | Expr::StateSelfRef(_) => Err(PrinterError::UnexpectedSelfRef),

            // A bare `BuiltinRef` in non-call position has no surface form: the
            // resolver only ever mints builtins as call callees (operators,
            // stdlib intrinsics). If one appears standalone we cannot invert it.
            Expr::BuiltinRef(name) => Err(PrinterError::UnknownBuiltin(format!(
                "{} (in non-call position)",
                name
            ))),

            Expr::Call(callee, args) => self.print_call(callee, args, binders, type_params, tp_names),

            Expr::Lambda { params, body } => {
                let mut out = String::from("|");
                let lam_names: Vec<String> = if self.local_names.active() {
                    (0..params.len()).map(|_| self.local_names.next()).collect()
                } else {
                    lambda_param_names(binders, params.len())
                };
                for (i, (pname, pty)) in lam_names.iter().zip(params.iter()).enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(pname);
                    out.push_str(": ");
                    out.push_str(&self.print_type(pty, type_params, tp_names)?);
                }
                out.push_str("| ");
                // Lambda params push binders, param 0 outermost (index
                // params.len()-1), param N-1 innermost (index 0) — matching the
                // resolver which pushes params in order.
                for pname in &lam_names {
                    binders.push(pname.clone());
                }
                let body_s = self.print_expr(body, binders, type_params, tp_names);
                for _ in 0..params.len() {
                    binders.pop();
                }
                out.push_str(&body_s?);
                Ok(out)
            }

            // `let`/`defer` chains lower from blocks. Re-render as a block so
            // the nesting + binder order matches the resolver exactly.
            Expr::Let { .. } | Expr::Defer { .. } => {
                self.print_block(e, binders, type_params, tp_names)
            }

            Expr::StructNew { struct_ref, fields } => {
                self.print_struct_new(struct_ref, fields, binders, type_params, tp_names)
            }

            Expr::Field {
                base,
                struct_ref,
                index,
            } => {
                let base_s = self.print_atom(base, binders, type_params, tp_names)?;
                let fname = self.struct_field_name(struct_ref, *index)?;
                Ok(format!("{}.{}", base_s, fname))
            }

            Expr::EnumNew {
                enum_ref,
                variant_index,
                payload,
            } => self.print_enum_new(enum_ref, *variant_index, payload, binders, type_params, tp_names),

            Expr::Match { scrutinee, arms } => {
                self.print_match(scrutinee, arms, binders, type_params, tp_names)
            }

            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let c = self.print_expr(cond, binders, type_params, tp_names)?;
                let t = self.print_expr(then_branch, binders, type_params, tp_names)?;
                let el = self.print_expr(else_branch, binders, type_params, tp_names)?;
                Ok(format!("if {} {{ {} }} else {{ {} }}", c, t, el))
            }

            Expr::Try {
                expr,
                enum_ref: _,
                ok_index: _,
                err_index: _,
            } => {
                // The resolver recovers enum_ref/ok/err structurally from the
                // operand's type, so the surface form is just `expr?`; the
                // bracketed indices are not written.
                let inner = self.print_atom(expr, binders, type_params, tp_names)?;
                Ok(format!("{}?", inner))
            }
        }
    }

    /// Print an expression that must bind tighter than an operator / postfix —
    /// wrap in parens when it isn't already atomic. Used for `?` operands,
    /// field bases, etc.
    fn print_atom(
        &self,
        e: &Expr,
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        let s = self.print_expr(e, binders, type_params, tp_names)?;
        if needs_parens_as_atom(e) {
            Ok(format!("({})", s))
        } else {
            Ok(s)
        }
    }

    fn print_call(
        &self,
        callee: &Expr,
        args: &[Expr],
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        // A builtin callee inverts to either an infix operator, a unary
        // operator, or a named stdlib-intrinsic call.
        if let Expr::BuiltinRef(name) = callee {
            return self.print_builtin_call(name, args, binders, type_params, tp_names);
        }
        // Ordinary call: `callee(arg, arg, ...)`.
        let callee_s = self.print_atom(callee, binders, type_params, tp_names)?;
        let mut parts = Vec::with_capacity(args.len());
        for a in args {
            parts.push(self.print_expr(a, binders, type_params, tp_names)?);
        }
        Ok(format!("{}({})", callee_s, parts.join(", ")))
    }

    fn print_builtin_call(
        &self,
        name: &str,
        args: &[Expr],
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        // Operators: invert the resolver's binop/unop tables.
        if let Some(op) = infix_op_for_builtin(name) {
            if args.len() != 2 {
                return Err(PrinterError::UnknownBuiltin(format!(
                    "{} expects 2 operands, got {}",
                    name,
                    args.len()
                )));
            }
            let l = self.print_operand(&args[0], binders, type_params, tp_names)?;
            let r = self.print_operand(&args[1], binders, type_params, tp_names)?;
            // Always parenthesise the whole binop so precedence + associativity
            // round-trip regardless of context.
            return Ok(format!("({} {} {})", l, op, r));
        }
        if let Some(op) = prefix_op_for_builtin(name) {
            if args.len() != 1 {
                return Err(PrinterError::UnknownBuiltin(format!(
                    "{} expects 1 operand, got {}",
                    name,
                    args.len()
                )));
            }
            let o = self.print_operand(&args[0], binders, type_params, tp_names)?;
            return Ok(format!("({}{})", op, o));
        }
        // Named stdlib intrinsics that the resolver lowers from a surface call
        // `foo(args)`. Invert to the same surface call.
        if let Some(surface) = surface_name_for_builtin(name) {
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(self.print_expr(a, binders, type_params, tp_names)?);
            }
            return Ok(format!("{}({})", surface, parts.join(", ")));
        }
        Err(PrinterError::UnknownBuiltin(name.to_owned()))
    }

    /// An operand of an infix/prefix operator. Parenthesisation is handled by
    /// the operator printer (it always wraps the whole op), so an operand only
    /// needs its own atom-level parens.
    fn print_operand(
        &self,
        e: &Expr,
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        self.print_expr(e, binders, type_params, tp_names)
    }

    fn print_struct_new(
        &self,
        struct_ref: &Hash,
        fields: &[Expr],
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        let def = self.cb.load_def(struct_ref)?;
        let field_names = match def {
            Def::Struct { fields: fs, .. } => fs,
            other => {
                return Err(PrinterError::BadTypeShape(format!(
                    "struct_ref {} is not a struct: {:?}",
                    struct_ref, other
                )));
            }
        };
        if field_names.len() != fields.len() {
            return Err(PrinterError::BadTypeShape(format!(
                "struct {} declares {} fields but constructor has {}",
                self.names.type_name(struct_ref),
                field_names.len(),
                fields.len()
            )));
        }
        let type_name = self.names.type_name(struct_ref);
        let mut parts = Vec::with_capacity(fields.len());
        for ((fname, _), fexpr) in field_names.iter().zip(fields.iter()) {
            let val = self.print_expr(fexpr, binders, type_params, tp_names)?;
            parts.push(format!("{}: {}", fname, val));
        }
        Ok(format!("{} {{ {} }}", type_name, parts.join(", ")))
    }

    fn print_enum_new(
        &self,
        enum_ref: &Hash,
        variant_index: u32,
        payload: &Option<Box<Expr>>,
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        let vname = self.enum_variant_name(enum_ref, variant_index)?;
        match payload {
            None => Ok(vname),
            Some(p) => {
                let val = self.print_expr(p, binders, type_params, tp_names)?;
                Ok(format!("{}({})", vname, val))
            }
        }
    }

    fn print_match(
        &self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        let scrut = self.print_expr(scrutinee, binders, type_params, tp_names)?;
        let mut out = format!("match {} {{ ", scrut);
        for (i, arm) in arms.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            // Render the pattern, collecting the binder names it introduces in
            // traversal order (matching the resolver's collect_pattern_bindings).
            let mut pat_binders: Vec<String> = Vec::new();
            let pat_s = self.print_pattern(&arm.pattern, &mut pat_binders)?;
            // Push pattern binders in the same order the resolver pushes them.
            for b in &pat_binders {
                binders.push(b.clone());
            }
            let body_s = self.print_expr(&arm.body, binders, type_params, tp_names);
            for _ in 0..pat_binders.len() {
                binders.pop();
            }
            out.push_str(&pat_s);
            out.push_str(" => ");
            out.push_str(&body_s?);
        }
        out.push_str(" }");
        Ok(out)
    }

    /// Render a pattern, appending each binding's chosen surface name to
    /// `binder_names` in traversal order. A `Pattern::Var` introduces exactly
    /// one binding; `Wildcard` and nullary enum patterns introduce none.
    fn print_pattern(
        &self,
        p: &Pattern,
        binder_names: &mut Vec<String>,
    ) -> Result<String, PrinterError> {
        match p {
            Pattern::Wildcard => Ok("_".to_owned()),
            Pattern::Var => {
                // The author's original name when the side-car is active,
                // else a deterministic, collision-free binding name keyed by
                // the binder's position within the arm. The actual de Bruijn
                // resolution happens via the binder stack; this name only has
                // to be a legal identifier that isn't a keyword and is unique
                // within the arm.
                let name = if self.local_names.active() {
                    self.local_names.next()
                } else {
                    format!("m{}", binder_names.len())
                };
                binder_names.push(name.clone());
                Ok(name)
            }
            Pattern::Enum {
                enum_ref,
                variant_index,
                payload,
            } => {
                let vname = self.enum_variant_name(enum_ref, *variant_index)?;
                match payload {
                    None => Ok(vname),
                    Some(sub) => {
                        let sub_s = self.print_pattern(sub, binder_names)?;
                        Ok(format!("{}({})", vname, sub_s))
                    }
                }
            }
        }
    }

    // ---- Block reconstruction (Let / Defer chains) ----

    /// Render a `Let`/`Defer` chain as a `{ ... }` block. The resolver lowers
    /// `{ let x = e1; defer c; tail }` to nested `Let`/`Defer` nodes; we invert
    /// that, pushing a `let` binder per `Let` (in source order) so the body's de
    /// Bruijn indices resolve correctly.
    fn print_block(
        &self,
        e: &Expr,
        binders: &mut Binders,
        type_params: u32,
        tp_names: &[String],
    ) -> Result<String, PrinterError> {
        let mut stmts: Vec<String> = Vec::new();
        let mut pushed = 0usize;
        let mut cur = e;
        // Walk the chain. Each `Let` introduces a binder; each `Defer` does not.
        loop {
            match cur {
                Expr::Let { value, body } => {
                    let val = self.print_expr(value, binders, type_params, tp_names)?;
                    let name = if self.local_names.active() {
                        self.local_names.next()
                    } else {
                        format!("v{}", binders.stack.len())
                    };
                    stmts.push(format!("let {} = {};", name, val));
                    binders.push(name);
                    pushed += 1;
                    cur = body;
                }
                Expr::Defer { cleanup, body } => {
                    // Cleanup is in the env at the defer point (no binder).
                    let val = self.print_expr(cleanup, binders, type_params, tp_names)?;
                    stmts.push(format!("defer {};", val));
                    cur = body;
                }
                _ => break,
            }
        }
        let tail = self.print_expr(cur, binders, type_params, tp_names);
        for _ in 0..pushed {
            binders.pop();
        }
        let tail = tail?;
        let mut out = String::from("{ ");
        for s in &stmts {
            out.push_str(s);
            out.push(' ');
        }
        out.push_str(&tail);
        out.push_str(" }");
        Ok(out)
    }

    // ---- Struct / enum metadata lookup ----

    fn struct_field_name(&self, struct_ref: &Hash, index: u32) -> Result<String, PrinterError> {
        let def = self.cb.load_def(struct_ref)?;
        match def {
            Def::Struct { fields, .. } => {
                fields
                    .get(index as usize)
                    .map(|(n, _)| n.clone())
                    .ok_or_else(|| {
                        PrinterError::BadTypeShape(format!(
                            "struct {} has no field at index {}",
                            self.names.type_name(struct_ref),
                            index
                        ))
                    })
            }
            other => Err(PrinterError::BadTypeShape(format!(
                "struct_ref {} is not a struct: {:?}",
                struct_ref, other
            ))),
        }
    }

    fn enum_variant_name(&self, enum_ref: &Hash, index: u32) -> Result<String, PrinterError> {
        let def = self.cb.load_def(enum_ref)?;
        match def {
            Def::Enum { variants, .. } => {
                // Variants are always printed qualified (`Enum::Variant`), the
                // only legal surface form.
                let enum_name = self.names.type_name(enum_ref);
                variants
                    .get(index as usize)
                    .map(|(n, _)| format!("{}::{}", enum_name, n))
                    .ok_or_else(|| {
                        PrinterError::BadTypeShape(format!(
                            "enum {} has no variant at index {}",
                            self.names.type_name(enum_ref),
                            index
                        ))
                    })
            }
            other => Err(PrinterError::BadTypeShape(format!(
                "enum_ref {} is not an enum: {:?}",
                enum_ref, other
            ))),
        }
    }
}

// =============================================================================
// Binder-name generators
// =============================================================================

/// Deterministic parameter names `p0, p1, ...` for a fn with `n` params.
fn param_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("p{}", i)).collect()
}

/// Deterministic lambda-parameter names that do not collide with any binder
/// currently in scope. Lambda params are named `lN_k` where `N` is the current
/// binder depth (unique per lambda nesting) and `k` is the param position.
fn lambda_param_names(binders: &Binders, n: usize) -> Vec<String> {
    let depth = binders.stack.len();
    (0..n).map(|k| format!("l{}_{}", depth, k)).collect()
}

/// Deterministic type-parameter names `T0, T1, ...`.
fn type_param_names(n: u32) -> Vec<String> {
    (0..n).map(|i| format!("T{}", i)).collect()
}

// =============================================================================
// Operator / builtin inversion (derived from resolve.rs)
// =============================================================================

/// Invert the resolver's `binop_builtin` / `binop_builtin_typed` tables: given
/// a `core/i64.*` or `core/f64.*` or `core/bool.*` builtin id minted for an
/// infix operator, return the surface operator string. Both the i64 and f64
/// variants map to the same surface operator (the resolver re-derives which by
/// operand type), so this is a faithful inverse.
fn infix_op_for_builtin(name: &str) -> Option<&'static str> {
    match name {
        "core/i64.add" | "core/f64.add" => Some("+"),
        "core/i64.sub" | "core/f64.sub" => Some("-"),
        "core/i64.mul" | "core/f64.mul" => Some("*"),
        "core/i64.div" | "core/f64.div" => Some("/"),
        "core/i64.rem" | "core/f64.rem" => Some("%"),
        "core/i64.eq" | "core/f64.eq" => Some("=="),
        "core/i64.ne" | "core/f64.ne" => Some("!="),
        "core/i64.lt" | "core/f64.lt" => Some("<"),
        "core/i64.le" | "core/f64.le" => Some("<="),
        "core/i64.gt" | "core/f64.gt" => Some(">"),
        "core/i64.ge" | "core/f64.ge" => Some(">="),
        "core/bool.and" => Some("&&"),
        "core/bool.or" => Some("||"),
        _ => None,
    }
}

/// Invert the resolver's `unop_builtin` table.
fn prefix_op_for_builtin(name: &str) -> Option<&'static str> {
    match name {
        "core/i64.neg" => Some("-"),
        "core/bool.not" => Some("!"),
        _ => None,
    }
}

/// Invert the resolver's named-intrinsic call lowering (the big `match
/// name.as_str()` in `resolve_expr_typed`'s Call branch, plus array builtins,
/// `gc_collect`, and `ext/<name>` externs). Returns the surface callee name.
fn surface_name_for_builtin(name: &str) -> Option<String> {
    // `ext/<name>` externs round-trip to a bare call on `<name>`.
    if let Some(ext) = name.strip_prefix("ext/") {
        return Some(ext.to_owned());
    }
    let s = match name {
        "core/string.len" => "string_len",
        "core/string.eq" => "string_eq",
        "core/string.concat" => "string_concat",
        "core/bytes.new" => "bytes_new",
        "core/bytes.len" => "bytes_len",
        "core/bytes.get" => "bytes_get_trusted",
        "core/bytes.set" => "bytes_set_trusted",
        "core/bytes.slice" => "bytes_slice",
        "core/bytes.concat" => "bytes_concat",
        "core/bytes.from_string" => "bytes_from_string",
        "core/string.from_bytes" => "string_from_bytes",
        "core/f64.of_int" => "int_to_float",
        "core/f64.to_int" => "float_to_int",
        "core/f64.sqrt" => "float_sqrt",
        "core/abort" => "abort",
        "core/ptr.null" => "ptr_null",
        "core/ptr.is_null" => "ptr_is_null",
        "core/ptr.add" => "ptr_add",
        "core/ptr.read_u8" => "ptr_read_u8",
        "core/ptr.write_u8" => "ptr_write_u8",
        "core/ptr.read_i64" => "ptr_read_i64",
        "core/ptr.write_i64" => "ptr_write_i64",
        "core/ptr.read_ptr" => "ptr_read_ptr",
        "core/ptr.write_ptr" => "ptr_write_ptr",
        "core/ptr.to_int" => "ptr_to_int",
        "core/ptr.from_int" => "int_to_ptr",
        "core/i64.and" => "bit_and",
        "core/i64.or" => "bit_or",
        "core/i64.xor" => "bit_xor",
        "core/i64.shl" => "bit_shl",
        "core/i64.shr" => "bit_shr",
        "core/array.new" => "array_new",
        "core/array.new_prim" => "array_new",
        "core/array.len" => "array_len",
        "core/array.get" => "array_get_trusted",
        "core/array.set" => "array_set_trusted",
        "core/gc.collect" => "gc_collect",
        _ => return None,
    };
    Some(s.to_owned())
}

// =============================================================================
// Literal rendering
// =============================================================================

/// Render an integer literal. The lexer (`lex_int`) only starts a numeric
/// literal on a digit: a leading `-` is the separate `Minus` operator, and the
/// parser turns `-n` into `Unary{Neg, IntLit(n)}`, which the resolver lowers to
/// `Call(BuiltinRef("core/i64.neg"), [IntLit(n)])` — a DIFFERENT canonical Expr
/// than `IntLit(-n)`. So a negative `IntLit` has no surface form that re-lexes
/// back to itself; printing `-n` would silently re-resolve to a Call (a
/// different hash). We therefore hard-error on negatives. (A real codebase
/// built from surface source never holds a negative `IntLit`; it holds the
/// neg-Call, which the operator printer inverts correctly to `(-n)`.)
fn render_int(v: i64) -> Result<String, PrinterError> {
    if v < 0 {
        return Err(PrinterError::Unrenderable(format!(
            "negative integer literal {} cannot be printed as a bare literal: the \
             lexer has no negative numeric literal (`-` is a separate operator), \
             so `{}` would re-resolve to Call(core/i64.neg, IntLit({})) — a \
             different canonical Expr",
            v,
            v,
            -v
        )));
    }
    Ok(v.to_string())
}

/// Render a float so it re-lexes to the exact same `f64`. The lexer's float
/// grammar (`lex_int`) accepts both a fractional part (`[0-9]+ "." [0-9]+`)
/// AND an exponent (`[eE][+-]?[0-9]+`), so Rust's shortest round-tripping `{}`
/// formatting is faithful for any non-negative finite value: it produces either
/// a dotted form, an exponent form, or an integer-valued form (which we pad with
/// `.0` so the lexer reads it as a Float, not an Int).
///
/// Genuine remaining gaps:
///   - Negative floats: the lexer has no negative numeric literal; `-1.5` lexes
///     as `Minus, Float(1.5)` and resolves to `Call(core/f64.neg, FloatLit)` —
///     a different canonical Expr. Hard error.
///   - Non-finite floats (NaN / ±Inf): no surface literal exists at all.
fn render_float(v: f64) -> Result<String, PrinterError> {
    if !v.is_finite() {
        return Err(PrinterError::Unrenderable(format!(
            "non-finite float {:?} has no surface literal",
            v
        )));
    }
    if v < 0.0 {
        // Negative floats would print as `-1.5`, which lexes as unary-neg of a
        // float literal — a DIFFERENT canonical Expr (Call(neg, FloatLit)).
        // Refuse rather than change the AST shape.
        return Err(PrinterError::Unrenderable(format!(
            "negative float literal {:?} cannot be printed as a bare literal: the \
             lexer has no negative numeric literal (`-` is a separate operator), \
             so it would re-resolve to Call(core/f64.neg, FloatLit) — a different \
             canonical Expr",
            v
        )));
    }
    let s = format!("{}", v);
    // Already a float to the lexer if it has a fractional part OR an exponent.
    if s.contains('.') || s.contains('e') || s.contains('E') {
        return Ok(s);
    }
    // Integer-valued form (e.g. `2` for 2.0, or `10000000000` for 1e10): the
    // lexer would read a bare digit run as an Int. Force a fractional part so it
    // lexes as a Float.
    Ok(format!("{}.0", s))
}

/// Render a string literal. The surface lexer reads `'"' ... '"'` and supports
/// the escape set `\n \r \t \\ \" \0 \e` (`\e` = ESC = U+001B); any other byte
/// between the quotes is taken verbatim via the lexer's UTF-8 decode path. So we
/// emit the standard escapes for the characters that have them (this keeps the
/// `"` / `\` delimiter-and-escape characters unambiguous and the result
/// readable) and emit every other character literally — which round-trips
/// byte-for-byte. Every `String` is therefore expressible; there is no
/// hard-error case here.
fn render_string(s: &str) -> Result<String, PrinterError> {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            '\u{1b}' => out.push_str("\\e"),
            other => out.push(other),
        }
    }
    out.push('"');
    Ok(out)
}

// =============================================================================
// Parenthesisation helper
// =============================================================================

/// Whether an expression must be wrapped in parens to be used as an "atom"
/// (the base of a field access, the operand of `?`, or a call callee). Literals,
/// variables, refs, already-parenthesised operator forms, calls, field
/// accesses, struct/enum constructors and `?` are atomic enough on their own.
fn needs_parens_as_atom(e: &Expr) -> bool {
    match e {
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::BuiltinRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::Field { .. }
        | Expr::Try { .. }
        | Expr::EnumNew { .. }
        | Expr::StructNew { .. } => false,
        // Calls print as `f(...)` / `op(...)` (operators are already
        // parenthesised by the operator printer), so they're atomic.
        Expr::Call(..) => false,
        // Control-flow / blocks / lambdas need wrapping when used as an atom.
        Expr::Lambda { .. }
        | Expr::Let { .. }
        | Expr::Defer { .. }
        | Expr::Match { .. }
        | Expr::If { .. } => true,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique temp dir under the OS temp dir (mirrors depindex's helper).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("ai_lang_printer_{}_{}_{}", pid, nanos, n));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Resolve `source` and store it into a fresh codebase. Returns the
    /// codebase (with names populated).
    fn build_codebase(tmp: &TempDir, source: &str) -> Codebase {
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(tmp.path()).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        // Capture + persist the author's local names so `view` can replay
        // them. This mirrors what a real edit/commit path would do.
        let names = crate::resolve::local_names_for_module(&module).expect("local names");
        cb.store_local_names_batch(&names).expect("store local names");
        cb
    }

    /// As `build_codebase` but WITHOUT persisting the local-name side-car,
    /// so the printer must fall back to `p0/p1/...`.
    fn build_codebase_no_names(tmp: &TempDir, source: &str) -> Codebase {
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(tmp.path()).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        cb
    }

    fn hash_of(cb: &Codebase, name: &str) -> Hash {
        cb.get_name(name)
            .unwrap_or_else(|| panic!("no name {:?} in codebase", name))
    }

    /// The round-trip property: print the def named `name`, re-parse +
    /// in the same context, and assert the printed def's hash equals the
    /// original. We re-resolve the WHOLE original source but with the chosen
    /// def's source replaced by its printed form (same name). Keeping the same
    /// name is load-bearing: a self-recursive def's body references its own
    /// name, so the recursion must resolve to itself (not to the un-replaced
    /// original) for the hash to match. Replacing in place achieves that while
    /// keeping every other referenced name/type/variant in scope.
    fn assert_round_trips(source: &str, name: &str) {
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, source);
        let orig = hash_of(&cb, name);

        let printed = print_def(&cb, orig).expect("print_def");

        let replaced = replace_def_source(source, name, &printed);
        let module = parse_module(&replaced).unwrap_or_else(|e| {
            panic!(
                "re-parse of printed def failed: {}\n--- printed ---\n{}\n--- replaced ---\n{}",
                e, printed, replaced
            )
        });
        let rm = resolve_module(&module).unwrap_or_else(|e| {
            panic!(
                "re-resolve of printed def failed: {}\n--- printed ---\n{}",
                e, printed
            )
        });
        let reprinted = rm.get(name).unwrap_or_else(|| {
            panic!("printed def vanished after re-resolve\nprinted: {}", printed)
        });
        assert_eq!(
            reprinted.hash, orig,
            "round-trip hash mismatch for `{}`\n--- printed ---\n{}",
            name, printed
        );
    }

    /// Build a module source where the def named `name` is replaced by
    /// `printed`, leaving every other def untouched. We re-resolve the original
    /// to find the def's surface boundaries by parsing, then splice the printed
    /// text in. Simpler + robust: parse the original into defs, drop the one
    /// named `name`, and append the printed replacement. Order doesn't matter
    /// for non-mutually-recursive defs in these tests (callees precede callers
    /// in the sources used), so we keep the others in order and put the printed
    /// def where it can see its dependencies: at the end if it only calls
    /// earlier defs, but for self-recursion it just needs its own name, which it
    /// has. To be safe against forward-reference rules, we reconstruct by
    /// textually replacing the original def's line(s).
    fn replace_def_source(source: &str, name: &str, printed: &str) -> String {
        // Re-emit the module: for every def in source order, print either the
        // original chunk or, for the target, the printed form. We slice the
        // source on top-level `def `/`struct `/`enum ` boundaries.
        let mut out = String::new();
        let chunks = split_top_level_defs(source);
        for chunk in chunks {
            let chunk_name = leading_def_name(&chunk);
            if chunk_name.as_deref() == Some(name) {
                out.push_str(printed);
            } else {
                out.push_str(&chunk);
            }
            out.push('\n');
        }
        out
    }

    /// Split a module source into per-definition chunks at top-level
    /// `def`/`struct`/`enum` keywords (column 0 after trimming leading
    /// whitespace on a line). Good enough for the self-contained test sources.
    fn split_top_level_defs(source: &str) -> Vec<String> {
        let mut chunks: Vec<String> = Vec::new();
        let mut cur = String::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let starts_def = trimmed.starts_with("def ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("enum ");
            if starts_def && !cur.trim().is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur.push_str(line);
            cur.push('\n');
        }
        if !cur.trim().is_empty() {
            chunks.push(cur);
        }
        chunks
    }

    /// Extract the name from a chunk whose first non-blank line starts with
    /// `def [local] <name>` / `struct <name>` / `enum <name>`.
    fn leading_def_name(chunk: &str) -> Option<String> {
        for line in chunk.lines() {
            let t = line.trim_start();
            let rest = t
                .strip_prefix("def local ")
                .or_else(|| t.strip_prefix("def "))
                .or_else(|| t.strip_prefix("struct "))
                .or_else(|| t.strip_prefix("enum "))?;
            // Name is up to the first non-identifier char.
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                return None;
            }
            return Some(name);
        }
        None
    }

    // ---- term (fn) round-trips ----

    #[test]
    fn factorial_recursive_round_trips() {
        let src = "
            def fact(n: Int) -> Int =
                if n == 0 { 1 } else { n * fact(n - 1) }
        ";
        assert_round_trips(src, "fact");
    }

    #[test]
    fn fn_calling_another_fn_round_trips() {
        let src = "
            def double(x: Int) -> Int = x * 2
            def quad(x: Int) -> Int = double(double(x))
        ";
        assert_round_trips(src, "quad");
    }

    #[test]
    fn if_else_round_trips() {
        let src = "
            def clamp(x: Int) -> Int = if x < 0 { 0 } else { x }
        ";
        assert_round_trips(src, "clamp");
    }

    #[test]
    fn let_block_round_trips() {
        let src = "
            def f(x: Int) -> Int = {
                let a = x + 1;
                let b = a * 2;
                a + b
            }
        ";
        assert_round_trips(src, "f");
    }

    #[test]
    fn lambda_round_trips() {
        let src = "
            def apply_twice(f: fn(Int) -> Int, x: Int) -> Int = f(f(x))
            def add_one_twice(x: Int) -> Int = apply_twice(|y: Int| y + 1, x)
        ";
        assert_round_trips(src, "add_one_twice");
    }

    #[test]
    fn struct_construction_and_field_access_round_trips() {
        let src = "
            struct Point { x: Int, y: Int }
            def make(a: Int, b: Int) -> Point = Point { x: a, y: b }
            def get_x(p: Point) -> Int = p.x
        ";
        assert_round_trips(src, "make");
        assert_round_trips(src, "get_x");
    }

    #[test]
    fn enum_match_and_construction_round_trips() {
        let src = "
            enum Color { Red, Green, Blue }
            def to_code(c: Color) -> Int =
                match c {
                    Color::Red => 0,
                    Color::Green => 1,
                    Color::Blue => 2
                }
            def pick() -> Color = Color::Green
        ";
        assert_round_trips(src, "to_code");
        assert_round_trips(src, "pick");
    }

    #[test]
    fn result_enum_with_payload_match_round_trips() {
        let src = "
            enum Res { Ok(Int), Err(Int) }
            def unwrap_or(r: Res, d: Int) -> Int =
                match r {
                    Res::Ok(v) => v,
                    Res::Err(e) => d
                }
            def mk_ok(n: Int) -> Res = Res::Ok(n)
        ";
        assert_round_trips(src, "unwrap_or");
        assert_round_trips(src, "mk_ok");
    }

    #[test]
    fn try_operator_round_trips() {
        // Uses the stdlib's Result<T,E> via `?` — same shape as try_demo.ail.
        let src = "
            enum Result<T, E> { Ok(T), Err(E) }
            def safe_div(a: Int, b: Int) -> Result<Int, Int> =
                if b == 0 { Result::Err(404) } else { Result::Ok(a / b) }
            def compute(x: Int, d: Int) -> Result<Int, Int> = {
                let q = safe_div(x, d)?;
                Result::Ok(q + 1)
            }
        ";
        assert_round_trips(src, "compute");
    }

    // ---- struct / enum DEF round-trips (compare against original name) ----

    fn assert_typedef_round_trips(source: &str, name: &str) {
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, source);
        let orig = hash_of(&cb, name);
        let printed = print_def(&cb, orig).expect("print_def");
        // Re-resolve the printed type def on its own — but its field/payload
        // types may reference other types in the module, so resolve the whole
        // module with this def replaced. Simplest faithful check: resolve the
        // ORIGINAL source (which contains this exact def) and confirm the
        // printed text parses+resolves to the same hash when substituted in.
        //
        // We re-parse just the printed def in a context that includes any other
        // type defs from the source by prepending them. For these tests the
        // type defs are self-contained, so resolving the printed def alone is
        // sufficient.
        let module = parse_module(&printed)
            .unwrap_or_else(|e| panic!("re-parse of printed type def failed: {}\n{}", e, printed));
        let rm = resolve_module(&module)
            .unwrap_or_else(|e| panic!("re-resolve of printed type def failed: {}\n{}", e, printed));
        let rd = rm
            .get(name)
            .unwrap_or_else(|| panic!("printed type def vanished: {}", printed));
        assert_eq!(
            rd.hash, orig,
            "type-def round-trip hash mismatch for `{}`\nprinted: {}",
            name, printed
        );
    }

    #[test]
    fn struct_def_round_trips() {
        let src = "struct Point { x: Int, y: Int }";
        assert_typedef_round_trips(src, "Point");
    }

    #[test]
    fn enum_def_round_trips() {
        let src = "enum Color { Red, Green, Blue }";
        assert_typedef_round_trips(src, "Color");
    }

    #[test]
    fn generic_enum_def_round_trips() {
        let src = "enum Opt<T> { Some(T), None }";
        assert_typedef_round_trips(src, "Opt");
    }

    // ---- literal round-trips (strings + floats) ----

    #[test]
    fn plain_string_round_trips() {
        let src = "def greeting() -> String = \"hello world\"";
        assert_round_trips(src, "greeting");
    }

    #[test]
    fn string_with_quotes_and_backslash_round_trips() {
        // Source contains an escaped quote and an escaped backslash. After
        // resolve the StringLit holds the raw chars; the printer must re-emit
        // the escapes so it re-lexes to the same bytes.
        let src = "def s() -> String = \"a\\\"b\\\\c\"";
        assert_round_trips(src, "s");
    }

    #[test]
    fn string_with_newline_tab_cr_round_trips() {
        let src = "def s() -> String = \"line1\\nline2\\tcol\\rend\"";
        assert_round_trips(src, "s");
    }

    #[test]
    fn string_with_nul_and_esc_round_trips() {
        let src = "def s() -> String = \"x\\0y\\ez\"";
        assert_round_trips(src, "s");
    }

    #[test]
    fn string_with_unicode_round_trips() {
        let src = "def s() -> String = \"café — π\"";
        assert_round_trips(src, "s");
    }

    #[test]
    fn float_literal_round_trips() {
        let src = "def pi() -> Float = 3.14";
        assert_round_trips(src, "pi");
    }

    #[test]
    fn integer_valued_float_round_trips() {
        let src = "def two() -> Float = 2.0";
        assert_round_trips(src, "two");
    }

    #[test]
    fn exponent_float_round_trips() {
        // `1e10` formats to `10000000000` (no dot); the printer must pad `.0`
        // so the lexer reads a Float, not an Int.
        let src = "def big() -> Float = 1e10";
        assert_round_trips(src, "big");
    }

    #[test]
    fn large_exponent_float_round_trips() {
        // `1e300` formats to `1e300` (exponent form, no dot): the lexer's float
        // grammar accepts the exponent, so it must be emitted as-is (NOT padded
        // with `.0`, which would be invalid).
        let src = "def huge() -> Float = 1e300";
        assert_round_trips(src, "huge");
    }

    // ---- inexpressible literals stay hard errors (grammar-driven) ----

    #[test]
    fn negative_int_literal_is_unrenderable() {
        // A bare negative IntLit has no surface form (`-` is a separate op).
        let e = render_int(-5).unwrap_err();
        assert!(matches!(e, PrinterError::Unrenderable(_)));
    }

    #[test]
    fn negative_float_literal_is_unrenderable() {
        let e = render_float(-1.5).unwrap_err();
        assert!(matches!(e, PrinterError::Unrenderable(_)));
    }

    #[test]
    fn non_finite_float_is_unrenderable() {
        assert!(matches!(
            render_float(f64::NAN).unwrap_err(),
            PrinterError::Unrenderable(_)
        ));
        assert!(matches!(
            render_float(f64::INFINITY).unwrap_err(),
            PrinterError::Unrenderable(_)
        ));
        assert!(matches!(
            render_float(f64::NEG_INFINITY).unwrap_err(),
            PrinterError::Unrenderable(_)
        ));
    }

    #[test]
    fn non_negative_int_and_float_render() {
        assert_eq!(render_int(0).unwrap(), "0");
        assert_eq!(render_int(42).unwrap(), "42");
        assert_eq!(render_float(2.0).unwrap(), "2.0");
        assert_eq!(render_float(3.14).unwrap(), "3.14");
        assert_eq!(render_float(1e10).unwrap(), "10000000000.0");
        // Whatever shape Rust's `{}` produces (Rust never emits `e` for f64, so
        // big integer-valued floats become a long digit run + `.0`), it must
        // re-parse to the exact same f64 via the lexer's float grammar.
        for v in [1e300_f64, 1e-300, 5e-324, 123456789012345.0] {
            let printed = render_float(v).unwrap();
            assert!(
                printed.contains('.') || printed.contains('e') || printed.contains('E'),
                "rendered float {:?} = {:?} would re-lex as an Int",
                v,
                printed
            );
            let back: f64 = printed.parse().unwrap();
            assert_eq!(back, v, "render_float({:?}) = {:?} did not round-trip", v, printed);
        }
    }

    // ---- synthetic alias for an un-named referenced hash ----

    #[test]
    fn unnamed_reference_uses_synthetic_alias() {
        // Build a codebase with a callee, then REMOVE its name so the caller's
        // TopRef has no name. The printer must emit `def_<8hex>` and not drop it.
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let mut cb = build_codebase(&tmp, src);
        let leaf = hash_of(&cb, "leaf");
        let caller = hash_of(&cb, "caller");
        cb.remove_name("leaf").expect("remove name");

        let printed = print_def(&cb, caller).expect("print_def");
        let alias = format!("def_{}", &leaf.to_hex()[..8]);
        assert!(
            printed.contains(&alias),
            "expected synthetic alias {} in printed output: {}",
            alias,
            printed
        );
    }

    // ---- local-name side-car: author names in `view` output ----

    #[test]
    fn params_use_author_names_and_round_trip() {
        let src = "def total(price: Int, qty: Int) -> Int = price * qty";
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, src);
        let h = hash_of(&cb, "total");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("price") && printed.contains("qty"),
            "expected author param names in output: {}",
            printed
        );
        assert!(
            !printed.contains("p0") && !printed.contains("p1"),
            "should not fall back to p0/p1 when names present: {}",
            printed
        );
        // The printed text (with real names) must STILL re-resolve to the
        // SAME hash — names never affect identity.
        assert_round_trips(src, "total");
    }

    #[test]
    fn fallback_to_p0_p1_when_sidecar_absent() {
        let src = "def total(price: Int, qty: Int) -> Int = price * qty";
        let tmp = TempDir::new();
        let cb = build_codebase_no_names(&tmp, src);
        let h = hash_of(&cb, "total");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("p0") && printed.contains("p1"),
            "expected p0/p1 fallback when no side-car: {}",
            printed
        );
        assert!(
            !printed.contains("price"),
            "must not invent author names without a side-car: {}",
            printed
        );
        // Fallback output still round-trips.
        assert_round_trips(src, "total");
    }

    #[test]
    fn let_binding_uses_author_name() {
        let src = "
            def f(x: Int) -> Int = {
                let subtotal = x + 1;
                subtotal * 2
            }
        ";
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, src);
        let h = hash_of(&cb, "f");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("subtotal"),
            "expected author let-name `subtotal` in output: {}",
            printed
        );
        assert_round_trips(src, "f");
    }

    #[test]
    fn lambda_param_uses_author_name() {
        let src = "
            def apply_twice(f: fn(Int) -> Int, x: Int) -> Int = f(f(x))
            def add_one_twice(x: Int) -> Int = apply_twice(|incr: Int| incr + 1, x)
        ";
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, src);
        let h = hash_of(&cb, "add_one_twice");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("incr"),
            "expected author lambda-param `incr` in output: {}",
            printed
        );
        assert_round_trips(src, "add_one_twice");
    }

    #[test]
    fn match_binding_uses_author_name() {
        let src = "
            enum Res { Ok(Int), Err(Int) }
            def unwrap_or(r: Res, d: Int) -> Int =
                match r {
                    Res::Ok(payload) => payload,
                    Res::Err(reason) => d
                }
        ";
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, src);
        let h = hash_of(&cb, "unwrap_or");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("payload") && printed.contains("reason"),
            "expected author match-binding names in output: {}",
            printed
        );
        assert_round_trips(src, "unwrap_or");
    }

    #[test]
    fn sidecar_persists_across_reopen() {
        let src = "def total(price: Int, qty: Int) -> Int = price * qty";
        let tmp = TempDir::new();
        let h = {
            let cb = build_codebase(&tmp, src);
            hash_of(&cb, "total")
        };
        // Reopen from disk: the side-car must still drive the names.
        let cb2 = Codebase::open(tmp.path()).expect("reopen");
        let printed = print_def(&cb2, h).expect("print_def");
        assert!(
            printed.contains("price") && printed.contains("qty"),
            "expected persisted author names after reopen: {}",
            printed
        );
    }

    #[test]
    fn stale_sidecar_wrong_arity_falls_back() {
        // A side-car whose length disagrees with the def's binder count is
        // ignored cleanly (the printer falls back to p0/p1), never rendering
        // names against the wrong binders.
        let src = "def total(price: Int, qty: Int) -> Int = price * qty";
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, src);
        let h = hash_of(&cb, "total");
        // Overwrite the side-car with the wrong number of names (1, not 2).
        cb.store_local_names(&h, &["only_one".to_owned()])
            .expect("store stale");
        let printed = print_def(&cb, h).expect("print_def");
        assert!(
            printed.contains("p0") && printed.contains("p1"),
            "expected p0/p1 fallback for arity-mismatched side-car: {}",
            printed
        );
        assert!(
            !printed.contains("only_one"),
            "must not use a stale, wrong-arity side-car: {}",
            printed
        );
    }
}
