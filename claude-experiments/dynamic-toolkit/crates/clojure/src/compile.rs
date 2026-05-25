//! Compiler: read values → dynir IR.
//!
//! Checkpoint 3 supports:
//!   - Self-evaluating literals
//!   - Local symbol lookup (params, `let`-bound vars)
//!   - Top-level fn lookup (everything in `func_refs`)
//!   - Special forms: `def`, `fn`, `if`, `let`, `do`, `quote`
//!   - Function calls (extern primitives + user-defined fns)
//!
//! `def` is recognized at top level and shaped specifically for
//! `(def NAME (fn [args...] body...))`. First-class fn values arrive
//! in a later checkpoint along with closures.

use std::collections::HashMap;

use dynir::builder::FunctionBuilder;
use dynir::ir::{BlockId, CmpOp, FuncRef, LiteralRef, Value};
use dynir::types::Type;
use dynlang::DynModule;
use dynlang::closure::{BodyShape, BoundBodyEnv, MakeClosure};
use dynlang::inline_body::InlineBody;
use dynlower::LiteralPool;

use crate::symbols::SymbolTable;
use crate::value as v;

/// One frame of the loop/recur target stack maintained while
/// lowering a function body. `recur` lowers to a jump to the most
/// recent target (a `loop` header, or — when no `loop` is open —
/// the function's entry block, which is what gives us tail-call
/// semantics for plain self-recursion).
///
/// `prepend` is non-empty for closure entry blocks: a closure's
/// entry block takes `(self_fn, p0, p1, …)`, and `recur` from inside
/// a closure body must pass the same self_fn through unchanged.
/// Parsed `(catch T name body...)` arm for `lower_try`.
///
///   - `type_sym = None`: catch-all (`_` or `:default`).
///   - `type_sym = Some(s)`: only matches a thrown record whose
///     `type_name` field equals symbol-id `s`.
///   - `bind_sym`: the local name the handler body uses for the
///     caught value.
///   - `body`: a list (possibly `nil`) of handler body forms.
struct CatchArm {
    type_sym: Option<u32>,
    bind_sym: u32,
    body: u64,
}

pub struct LoopTarget {
    pub block: BlockId,
    /// Number of user-visible recur arguments. For variadic targets
    /// this is `min_arity + 1` (the rest binding counts as one).
    pub arity: usize,
    /// Values to prepend to the recur args when emitting the jump.
    pub prepend: Vec<Value>,
    /// If true, the recur args must be packed into a single Clojure
    /// list before the jump (the target block's last param is an
    /// args_list — used by closures and variadic def-fns whose body
    /// uses the single-list ABI). If false, recur args go to the
    /// target's block params one-to-one (used by `loop` forms and
    /// fixed-arity def-fns whose body uses direct args).
    pub pack_args: bool,
}

/// Metadata for everything looked up by name in `func_refs`.
///
/// Two call shapes coexist:
///
/// - `Extern` — a C extern called with its natural N-argument
///   signature.
/// - `DefFn` — a user-defined fn introduced by `def`/`defmacro`. The
///   body has the uniform single-list ABI: signature `(args_list)`.
///   The body's prologue dispatches across the fn's arity clauses
///   (single-arity is just a 1-clause case) and unpacks
///   `args_list` to bind params. Static call sites pack args into a
///   list at compile time.
///
/// Closures (anonymous `(fn …)` expressions) are NOT in this table
/// — they have no source-level name and are only ever called via
/// the indirect path, which uses `Fn.func_ref` directly. Closures
/// have signature `(self_fn, args_list)` to give the body access to
/// captured values.
#[derive(Clone, Copy)]
pub enum FnEntry {
    Extern { fref: FuncRef, arity: usize },
    DefFn { fref: FuncRef },
}

impl FnEntry {
    pub fn fref(self) -> FuncRef {
        match self {
            FnEntry::Extern { fref, .. } => fref,
            FnEntry::DefFn { fref, .. } => fref,
        }
    }
}

/// Pre-resolved FuncRefs for every extern the lowering pipeline calls
/// by name. Resolved once at engine init from `func_refs`; held by
/// `Compiler` for the lifetime of the engine.
///
/// Hot-path benefit is incidental — the real win is readability at
/// every call site. Compare:
///
/// ```ignore
/// let fref = self.func_refs.get("__cons").copied()
///     .expect("__cons extern not registered").fref();
/// fb.call(fref, &[head, tail]).expect("__cons returns a value")
/// ```
///
/// to:
///
/// ```ignore
/// fb.call(self.externs.cons, &[head, tail])
///     .expect("__cons returns a value")
/// ```
///
/// New externs added here must also be registered in
/// [`crate::externs::all_prims`] (which is what populates the
/// `func_refs` map this struct is resolved from).
#[derive(Clone, Copy)]
pub struct Externs {
    pub alloc_record: FuncRef,
    pub arity_check: FuncRef,
    pub cons: FuncRef,
    pub def_value: FuncRef,
    pub exception_type_matches: FuncRef,
    pub method_lookup: FuncRef,
    pub raise_exception: FuncRef,
    pub reader_list_first: FuncRef,
    pub record_get_field: FuncRef,
    pub record_set_field: FuncRef,
    pub register_deftype: FuncRef,
    pub register_method: FuncRef,
    pub register_protocol_member: FuncRef,
    pub vector_from_list: FuncRef,
}

impl Externs {
    /// Resolve every required extern from `func_refs`. Panics with a
    /// clear message naming the missing extern if any aren't
    /// registered — that's an engine-init bug (`externs::all_prims`
    /// out of sync with this struct), not user error.
    pub fn resolve(func_refs: &HashMap<String, FnEntry>) -> Self {
        let lookup = |name: &str| -> FuncRef {
            match func_refs.get(name) {
                Some(FnEntry::Extern { fref, .. }) => *fref,
                Some(FnEntry::DefFn { .. }) => panic!(
                    "Externs::resolve: '{name}' is registered as a def-fn, \
                     not an extern — probably means a user `(def {name} …)` \
                     shadowed the runtime primitive"
                ),
                None => panic!(
                    "Externs::resolve: extern '{name}' not declared by \
                     `externs::all_prims()` — Externs struct out of sync \
                     with the prim registry"
                ),
            }
        };
        Externs {
            alloc_record: lookup("__alloc_record"),
            arity_check: lookup("__arity_check"),
            cons: lookup("__cons"),
            def_value: lookup("__def_value"),
            exception_type_matches: lookup("__exception_type_matches"),
            method_lookup: lookup("__method_lookup"),
            raise_exception: lookup("__raise_exception"),
            reader_list_first: lookup("__reader_list_first"),
            record_get_field: lookup("__record_get_field"),
            record_set_field: lookup("__record_set_field"),
            register_deftype: lookup("__register_deftype"),
            register_method: lookup("__register_method"),
            register_protocol_member: lookup("__register_protocol_member"),
            vector_from_list: lookup("__vector_from_list"),
        }
    }
}

/// Parsed parameter list. `(fn [a b & rs] …)` produces
/// `Params { fixed: [a, b], rest: Some(rs) }`. `min_arity` is the
/// number of fixed params; the user must supply at least that many
/// at the call site, with any extras packed into the rest list.
#[derive(Default)]
pub struct Params {
    pub fixed: Vec<u32>,
    pub rest: Option<u32>,
}

impl Params {
    pub fn min_arity(&self) -> usize {
        self.fixed.len()
    }
    pub fn is_variadic(&self) -> bool {
        self.rest.is_some()
    }
}

/// Parse a `[a b & rs]`-style parameter vector (or list) into a
/// `Params` struct. Panics on malformed input — every element must
/// be a symbol; `&` may appear at most once and must be followed by
/// exactly one symbol (the rest binding).
fn parse_params(arg_vec: u64, sym: &crate::symbols::SymbolTable) -> Params {
    let mut p = Params::default();
    let mut seen_amp = false;
    for elem in crate::collections::seq_iter(arg_vec) {
        if !v::is_sym_id(elem) {
            panic!("fn: parameter must be a symbol");
        }
        let id = v::as_sym_id(elem);
        let name = sym.name(id);
        if name == "&" {
            if seen_amp {
                panic!("fn: only one `&` allowed in parameter list");
            }
            seen_amp = true;
            continue;
        }
        if seen_amp {
            if p.rest.is_some() {
                panic!("fn: only one binding allowed after `&`");
            }
            p.rest = Some(id);
        } else {
            p.fixed.push(id);
        }
    }
    if seen_amp && p.rest.is_none() {
        panic!("fn: `&` must be followed by a rest-binding symbol");
    }
    p
}

/// Encode (min_arity, is_variadic) into the single Raw64 word stored
/// in `Fn.arity`. Bit 63 is the variadic flag; the low 63 bits are
/// `min_arity`. Decoded by the runtime arity-check / dispatch path.
const ARITY_VARIADIC_BIT: u64 = 1 << 63;
pub fn encode_arity(min_arity: usize, is_variadic: bool) -> u64 {
    let n = min_arity as u64;
    if is_variadic {
        n | ARITY_VARIADIC_BIT
    } else {
        n
    }
}

/// One clause of a (possibly multi-arity) fn.
pub struct Clause {
    pub params: Params,
    pub body: u64,
}

/// Parse the rest of a `fn`-form (whatever follows the `fn` symbol)
/// into a vector of clauses. Two source shapes:
///   - Single-arity: `(fn [params] body…)` — `fn_rest`'s first
///     element is a vector. Returns one clause.
///   - Multi-arity:  `(fn ([params] body…) ([params] body…) …)` —
///     `fn_rest`'s first element is a list. Each list-element is
///     itself a `([params] body…)` clause.
fn parse_clauses(fn_rest: u64, sym: &crate::symbols::SymbolTable) -> Vec<Clause> {
    let head = v::first(fn_rest);
    if v::is_ptr(head) && crate::collections::is_list(head) {
        // Multi-arity. Each element of fn_rest is `([params] body…)`.
        v::list_iter(fn_rest)
            .map(|clause_form| {
                let arg_vec = v::first(clause_form);
                let body = v::rest(clause_form);
                Clause {
                    params: parse_params(arg_vec, sym),
                    body,
                }
            })
            .collect()
    } else {
        // Single-arity.
        let arg_vec = head;
        let body = v::rest(fn_rest);
        vec![Clause {
            params: parse_params(arg_vec, sym),
            body,
        }]
    }
}

#[derive(Debug)]
pub enum TopResult {
    /// Anonymous top-level expression compiled to a 0-arg function
    /// returning the result bits.
    Expr(FuncRef),
    /// `(def name VALUE)` where VALUE is not a `(fn …)` literal.
    /// `value_thunk` is a 0-arg fn that, when called, returns the
    /// evaluated value. The driver runs the thunk, takes its
    /// result, and interns a Var named `name` in `clojure.core`
    /// whose root is that result.
    DefineValue { name: String, value_thunk: FuncRef },
    /// `(def name ...)` (or `defmacro`) declared a new top-level
    /// function. There's no expression to run; the driver interns a
    /// Var in `clojure.core` whose root holds an `Fn` heap object
    /// pointing at this `FuncRef`. When `is_macro` is true, the
    /// driver also sets the `:macro` flag bit on the Var.
    /// `arity_word` is the already-encoded arity (low bits = min_arity,
    /// bit 63 = variadic flag). Multi-arity defns set variadic so the
    /// call-site arity_check accepts any `n >= min` and the body's
    /// dispatch validates exact arity.
    Define {
        name: String,
        fref: FuncRef,
        arity_word: u64,
        is_macro: bool,
    },
    /// Pure no-op (e.g. comment-only input).
    None,
}

pub struct Compiler<'a> {
    /// The toolkit module — owns the `ModuleBuilder` we declare/define
    /// every Clojure function on, plus the obj-type registry that
    /// ClosureKit and friends consult. Reach in via
    /// `self.dyn_module.module_builder` for raw IR operations.
    pub dyn_module: &'a mut DynModule,
    pub func_refs: &'a mut HashMap<String, FnEntry>,
    /// Pre-resolved FuncRefs for every extern the lowering pipeline
    /// calls by name. See [`Externs`].
    pub externs: &'a Externs,
    pub sym: &'a SymbolTable,
    pub anon_counter: &'a mut u32,
    /// Literal pool we push GC-managed compile-time constants into.
    /// Each push returns a slot index that becomes a `LiteralRef` in
    /// emitted IR; the JIT lowers `Inst::GcLiteral(idx)` to a load
    /// from `pool.base() + idx*8`. The pool is registered with the
    /// GC as a root source so moving collections rewrite the slots in
    /// place.
    pub literal_pool: &'a LiteralPool,
    /// Stable base address of the JitModule's `call_table`. Used by
    /// the indirect-call path to translate a runtime `FuncRef` index
    /// (read from a heap `Fn`) into the code pointer to call:
    /// `code_ptr = *(call_table_base + fr * 8)`.
    pub call_table_base: u64,
    /// Stack of recur targets. Pushed on entry to a `loop` form or a
    /// function body; popped on exit. The compiler is reset between
    /// top-level forms so this only ever grows during a single
    /// function compilation.
    pub loop_targets: Vec<LoopTarget>,
    /// Set to `true` by `lower_throw` / `lower_recur` after they
    /// terminate the current block and switch to a dead block.
    /// Cleared at the start of every `lower_expr` call so it tracks
    /// only the most-recent expression's exit shape.
    pub last_expr_non_returning: bool,
    /// The `clojure.core` namespace pointer, threaded through so the
    /// compiler can look up Vars at compile time. A bare symbol that
    /// names a `def`d fn resolves through that Var: we push the Var
    /// ptr into the literal pool and emit a load of `Var.root`,
    /// which holds the current `Fn` heap obj. Fresh on each `eval`
    /// driver run; the Var ptr is stable once interned.
    pub core_ns: u64,
    /// Compile-time deftype field map (`type-name sym → field syms`).
    /// Lives in `CompileState` so it survives across forms.
    /// `compile_deftype` writes; `compile_extend_type` reads to wrap
    /// each method body in implicit field-binding lets.
    pub deftype_fields: &'a mut HashMap<u32, Vec<u32>>,
    /// Toolkit closure primitive — owns the body-declaration / prologue
    /// / alloc / indirect-call IR. Every closure-related lowering site
    /// goes through this instead of hand-rolled IR.
    pub closures: &'a dynlang::closure::ClosureKit,
}

impl<'a> Compiler<'a> {
    pub fn compile_top(&mut self, form: u64) -> TopResult {
        // Recognize `(def NAME EXPR)` and `(defmacro NAME [args] body)`
        // at top level. Only lists carry a head — vectors/maps/strings
        // would read garbage through `v::first` since they share the
        // ptr tag but not the list layout.
        if v::is_ptr(form) && crate::collections::is_list(form) {
            let head = v::first(form);
            if v::is_sym_id(head) {
                let head_name = self.sym.name(v::as_sym_id(head)).to_string();
                match head_name.as_str() {
                    "def" => return self.compile_def(form),
                    "defmacro" => return self.compile_defmacro(form),
                    "deftype*" => return self.compile_deftype(form),
                    // `(deftype Name [fields] Proto1 (m1 …) Proto2 (m2 …))`
                    // — combined deftype* + extend-type. Compile both
                    // pieces sequentially.
                    "deftype" => return self.compile_deftype_combined(form),
                    "defprotocol" => return self.compile_defprotocol(form),
                    "extend-type" => return self.compile_extend_type(form),
                    // `(ns clojure.core)` and friends — for now we
                    // create `clojure.core` at engine startup and
                    // there's no real namespace switching, so just
                    // accept and ignore the directive.
                    "ns" | "in-ns" => return TopResult::None,
                    // `(defn name "doc"? {meta}? args-or-clauses…)`
                    // → `(def name (fn args-or-clauses…))`. Treated
                    // as a compile-time rewrite. `defn-` is the same
                    // (we don't enforce visibility yet).
                    "defn" | "defn-" => return self.compile_defn(form),
                    // `(declare name1 name2 …)` interns each name as
                    // an unbound Var in clojure.core so subsequent
                    // forms can reference them. The Var.root stays
                    // NIL until a real `def`/`defn` overwrites it via
                    // ns_intern's find-or-update. Required ahead of
                    // any forward reference: without a Var at compile
                    // time the symbol is genuinely undefined and
                    // lower_expr errors.
                    "declare" => return self.compile_declare(form),
                    _ => {}
                }
            }
        }
        // Otherwise: anonymous top-level expression.
        *self.anon_counter += 1;
        let name = format!("__top_{}", *self.anon_counter);
        let fref = self.dyn_module.module_builder.declare_func(&name, &[], Some(Type::I64));
        // Top-level exprs aren't callable by name — they're invoked
        // directly by `Engine::eval` via `run_jit(fref, &[], …)`.
        // No `func_refs` entry needed.

        let mut fb = self.dyn_module.module_builder.define_func(fref);
        let mut env = Env::new();
        let result = self.lower_expr(&mut fb, &mut env, form);
        fb.ret(result);
        self.dyn_module.module_builder.finish_func(fref, fb);

        TopResult::Expr(fref)
    }

    fn compile_declare(&mut self, form: u64) -> TopResult {
        // (declare a b c …) — intern each as an unbound Var (root NIL)
        // in clojure.core. Idempotent: if a Var already exists for the
        // name, ns_intern overwrites its root with NIL, which is wrong
        // if the Var was already bound to something useful. Guard the
        // overwrite: only intern fresh names.
        for name_v in crate::collections::seq_iter(v::rest(form)) {
            if !v::is_sym_id(name_v) {
                panic!("declare: each name must be a symbol");
            }
            let sym_value = name_v;
            let existing = crate::namespace::ns_lookup(self.core_ns, sym_value);
            if v::is_ptr(existing) {
                continue;
            }
            dynobj::roots::with_scope(64, |scope| {
                let _ = crate::namespace::ns_intern(
                    scope,
                    self.core_ns,
                    sym_value,
                    v::NIL,
                );
            });
        }
        TopResult::None
    }

    fn compile_def(&mut self, form: u64) -> TopResult {
        // `(def NAME VALUE)`. If VALUE is `(fn …)` we use the
        // ordinary fn-as-def path (registers a Var whose root is
        // an Fn obj). Otherwise compile a 0-arg thunk that
        // evaluates VALUE; the driver runs it and interns a Var
        // with the result as its root.
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("def: name must be a symbol");
        }
        let name = self.sym.name(v::as_sym_id(name_v)).to_string();
        let expr = v::first(v::rest(rest));

        let is_fn_literal = if v::is_ptr(expr) && crate::collections::is_list(expr) {
            let head = v::first(expr);
            v::is_sym_id(head) && self.sym.name(v::as_sym_id(head)) == "fn"
        } else {
            false
        };

        if is_fn_literal {
            let (name, fref, arity_word) =
                self.compile_def_like(form, "def", /*expect_fn=*/ true);
            return TopResult::Define {
                name,
                fref,
                arity_word,
                is_macro: false,
            };
        }

        // Non-fn def. Compile a 0-arg thunk that evaluates `expr`.
        *self.anon_counter += 1;
        let thunk_name = format!("__def_value_thunk_{}", *self.anon_counter);
        let thunk_fref = self.dyn_module.module_builder.declare_func(&thunk_name, &[], Some(Type::I64));
        let mut fb = self.dyn_module.module_builder.define_func(thunk_fref);
        let mut env = Env::new();
        let result = self.lower_expr(&mut fb, &mut env, expr);
        fb.ret(result);
        self.dyn_module.module_builder.finish_func(thunk_fref, fb);
        TopResult::DefineValue {
            name,
            value_thunk: thunk_fref,
        }
    }

    fn compile_defmacro(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("defmacro: name must be a symbol");
        }
        let name = self.sym.name(v::as_sym_id(name_v)).to_string();
        let mut after_name = v::rest(rest);

        // Skip optional docstring (a String literal) and metadata
        // map, mirroring compile_defn so user macros can carry the
        // same `(defmacro name "doc" {meta} [params] body)` shape.
        let first_after = v::first(after_name);
        if crate::collections::is_string(first_after) {
            after_name = v::rest(after_name);
        }
        let first_after2 = v::first(after_name);
        if crate::collections::is_map(first_after2) {
            after_name = v::rest(after_name);
        }

        // Same single-arity / multi-arity split as defn: peek at the
        // first element after the name. Vector → single arity, list →
        // multi-arity clauses.
        let mut clauses = parse_clauses(after_name, self.sym);

        // Match Clojure's defmacro: implicit &form / &env params get
        // prepended to whatever the user wrote, in EVERY arity.
        // expand.rs supplies both when invoking the macro fn.
        let form_id = self.sym.intern("&form");
        let env_id = self.sym.intern("&env");
        for c in clauses.iter_mut() {
            let mut fixed = Vec::with_capacity(c.params.fixed.len() + 2);
            fixed.push(form_id);
            fixed.push(env_id);
            fixed.extend_from_slice(&c.params.fixed);
            c.params.fixed = fixed;
        }

        let multi = clauses.len() > 1;
        let (fref, arity, is_variadic) = self.compile_def_fn(&name, clauses);
        let arity_word = encode_arity(arity, is_variadic || multi);
        TopResult::Define {
            name,
            fref,
            arity_word,
            is_macro: true,
        }
    }

    /// `(defn name docstring? metadata? body…)` →
    /// `(def name (fn body…))`. Strips an optional docstring (a
    /// string literal as the first body element) and an optional
    /// metadata map (a map literal as the next body element).
    fn compile_defn(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        let mut after_name = v::rest(rest);

        // Skip optional docstring (a String literal).
        let first_after = v::first(after_name);
        if crate::collections::is_string(first_after) {
            after_name = v::rest(after_name);
        }

        // Skip optional metadata map.
        let first_after2 = v::first(after_name);
        if crate::collections::is_map(first_after2) {
            after_name = v::rest(after_name);
        }

        // Synthesize (def name (fn after_name…)).
        let fn_sym_id = self.sym.intern("fn");
        let def_sym_id = self.sym.intern("def");
        let fn_sym = v::encode_sym_id(fn_sym_id);
        let def_sym = v::encode_sym_id(def_sym_id);

        // (fn . after_name)
        let fn_form = dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, fn_sym, after_name).get()
        });
        // (name fn_form . nil)
        let after_def = dynobj::roots::with_scope(8, |scope| {
            let tail = v::alloc_list_cell_from_raw(scope, fn_form, v::NIL);
            v::alloc_list_cell_from_raw(scope, name_v, tail.get()).get()
        });
        // (def . after_def)
        let synthesized = dynobj::roots::with_scope(8, |scope| {
            v::alloc_list_cell_from_raw(scope, def_sym, after_def).get()
        });
        self.compile_def(synthesized)
    }

    /// `(deftype Name [fields] Proto (m1 …) (m2 …) Proto2 (…) …)`
    ///
    /// Splits into a `deftype*` (just the name + fields) and an
    /// `extend-type` over the remaining body. Compiles both as a
    /// `do`-style sequence by emitting one of each and binding the
    /// type's value into the namespace.
    fn compile_deftype_combined(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        let after_name = v::rest(rest);
        let fields_v = v::first(after_name);
        let body_after_fields = v::rest(after_name);

        // Build `(deftype* Name fields)`.
        let deftype_star_sym = v::encode_sym_id(self.sym.intern("deftype*"));
        let dt_form = dynobj::roots::with_scope(32, |scope| {
            let tail = v::alloc_list_cell_from_raw(scope, fields_v, v::NIL);
            let mid = v::alloc_list_cell_from_raw(scope, name_v, tail.get());
            v::alloc_list_cell_from_raw(scope, deftype_star_sym, mid.get()).get()
        });
        let dt_result = self.compile_deftype(dt_form);

        if v::is_nil(body_after_fields) {
            return dt_result;
        }

        // Build `(extend-type Name body…)`.
        let ext_sym = v::encode_sym_id(self.sym.intern("extend-type"));
        let ext_form = dynobj::roots::with_scope(32, |scope| {
            let after = v::alloc_list_cell_from_raw(scope, name_v, body_after_fields);
            v::alloc_list_cell_from_raw(scope, ext_sym, after.get()).get()
        });
        let ext_result = self.compile_extend_type(ext_form);

        // Both compile_deftype and compile_extend_type return a thunk
        // (TopResult::Expr) that needs to RUN to install runtime
        // metadata: the deftype thunk fills `host.deftype_fields` so
        // `(.-field receiver)` can find the field's index, and the
        // extend thunk wires up the protocol method table. We can't
        // discard either — but the eval driver only runs one thunk
        // per TopResult. Combine them here into a single `(do dt ext)`
        // wrapper thunk that calls both in order.
        let (dt_fref, ext_fref) = match (dt_result, ext_result) {
            (TopResult::Expr(d), TopResult::Expr(e)) => (d, e),
            // If either side wasn't a thunk (e.g. extend-type body
            // empty), fall back to whichever runs.
            (TopResult::Expr(d), TopResult::None) => return TopResult::Expr(d),
            (TopResult::None, TopResult::Expr(e)) => return TopResult::Expr(e),
            _ => return TopResult::None,
        };

        *self.anon_counter += 1;
        let combined_name =
            format!("__deftype_combined_{}", *self.anon_counter);
        let combined_fref =
            self.dyn_module.module_builder.declare_func(&combined_name, &[], Some(Type::I64));
        {
            let mut fb = self.dyn_module.module_builder.define_func(combined_fref);
            fb.safepoint(&[]);
            fb.call(dt_fref, &[]);
            fb.safepoint(&[]);
            fb.call(ext_fref, &[]);
            let nil_v = fb.iconst(Type::I64, v::NIL as i64);
            fb.ret(nil_v);
            self.dyn_module.module_builder.finish_func(combined_fref, fb);
        }
        TopResult::Expr(combined_fref)
    }

    /// `(deftype* Name [field0 field1 …])`
    ///
    /// Compiles to a top-level expression that, when run, registers
    /// the type's field metadata with the host. The constructor
    /// `Name.` is also defined as a regular def-fn that allocates a
    /// `Record` and copies fields from its args.
    ///
    /// The returned `TopResult::Expr` runs a registration thunk —
    /// the driver invokes it inline after compilation, the same way
    /// it runs anonymous top-level expressions.
    fn compile_deftype(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("deftype*: name must be a symbol");
        }
        let name_id = v::as_sym_id(name_v);
        let name = self.sym.name(name_id).to_string();

        let fields_form = v::first(v::rest(rest));
        let mut field_ids: Vec<u32> = Vec::new();
        for f in crate::collections::seq_iter(fields_form) {
            if !v::is_sym_id(f) {
                panic!("deftype*: each field must be a symbol");
            }
            field_ids.push(v::as_sym_id(f));
        }
        let n_fields = field_ids.len();
        // Compile-time mirror of the runtime `deftype_fields`. Lets
        // `compile_extend_type` resolve `T`'s field names without
        // waiting for the runtime registration thunk.
        self.deftype_fields.insert(name_id, field_ids.clone());

        // Generate the constructor `Name.` as a regular def-fn. The
        // body uses the closure-kit prologue (args-list ABI, no
        // captures) to bind each field, then allocates a Record from
        // the bound values.
        let ctor_name = format!("{}.", name);
        let ctor_shape = BodyShape {
            fixed: n_fields,
            variadic: false,
            n_captures: 0,
        };
        let ctor_fref = self.closures.declare_body(
            &mut self.dyn_module.module_builder,
            &ctor_name,
            ctor_shape,
        );
        self.func_refs
            .insert(ctor_name, FnEntry::DefFn { fref: ctor_fref });

        // Body: prologue (arity check + args-list walk → field values),
        // then spill fields into a buffer and call __alloc_record.
        {
            let mut fb = self.dyn_module.module_builder.define_func(ctor_fref);
            let bound = self.closures.begin_body(&mut fb, ctor_shape);
            let field_vals = bound.args;

            // Spill fields to a stack buffer for __alloc_record.
            let buf_slot = fb.create_stack_slot((n_fields * 8) as u32, /*is_gc_root=*/ true);
            let buf_addr = fb.stack_addr(buf_slot);
            for (i, fv) in field_vals.iter().enumerate() {
                fb.store(*fv, buf_addr, (i * 8) as i32);
            }
            let buf_addr_i64 = fb.bitcast(buf_addr, Type::I64);

            let type_name_const = fb.iconst(Type::I64, v::encode_sym_id(name_id) as i64);
            let n_const = fb.iconst(Type::I64, n_fields as i64);

            let mut live = field_vals.clone();
            live.push(buf_addr_i64);
            fb.safepoint(&live);
            let result = fb
                .call(self.externs.alloc_record, &[type_name_const, n_const, buf_addr_i64])
                .expect("__alloc_record returns a value");
            fb.ret(result);
            self.dyn_module.module_builder.finish_func(ctor_fref, fb);
        }

        // Top-level registration thunk: call __register_deftype with
        // the type-name and field-name list.
        *self.anon_counter += 1;
        let reg_name = format!("__register_deftype_{}", *self.anon_counter);
        let reg_fref = self.dyn_module.module_builder.declare_func(&reg_name, &[], Some(Type::I64));
        {
            let mut fb = self.dyn_module.module_builder.define_func(reg_fref);
            // Spill field-name sym values to a buffer.
            let buf_slot = fb.create_stack_slot((n_fields.max(1) * 8) as u32, /*is_gc_root=*/ false);
            let buf_addr = fb.stack_addr(buf_slot);
            for (i, &fid) in field_ids.iter().enumerate() {
                let fv = fb.iconst(Type::I64, v::encode_sym_id(fid) as i64);
                fb.store(fv, buf_addr, (i * 8) as i32);
            }
            let buf_addr_i64 = fb.bitcast(buf_addr, Type::I64);
            let type_name_const = fb.iconst(Type::I64, v::encode_sym_id(name_id) as i64);
            let n_const = fb.iconst(Type::I64, n_fields as i64);
            fb.safepoint(&[]);
            fb.call(self.externs.register_deftype, &[type_name_const, n_const, buf_addr_i64]);

            // Also bind a Var `Name` whose root is the symbol
            // `'Name` so bare references to the type name (e.g.
            // `(instance? Reduced x)`) resolve to a value.
            fb.safepoint(&[]);
            fb.call(self.externs.def_value, &[type_name_const, type_name_const]);

            let nil_v = fb.iconst(Type::I64, v::NIL as i64);
            fb.ret(nil_v);
            self.dyn_module.module_builder.finish_func(reg_fref, fb);
        }
        TopResult::Expr(reg_fref)
    }

    /// `(defprotocol Name docstring? (m1 [args]...) (m2 [args]...))`
    ///
    /// For each method, generate a `def`d top-level fn that
    /// dispatches on the receiver's type via `__method_lookup` and
    /// invokes the matching `Fn` heap obj. This makes bare uses
    /// like `(-first xs)` work — they call the wrapper fn, which
    /// performs the dispatch.
    ///
    /// The wrapper's signature is variadic `[& args]`: a single
    /// args_list parameter, with the first element treated as the
    /// receiver. Multi-arity method declarations in the protocol
    /// are accepted but not separately compiled — the variadic
    /// wrapper handles every arity uniformly.
    fn compile_defprotocol(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let proto_name_v = v::first(rest);
        let proto_sym_id = if v::is_sym_id(proto_name_v) {
            v::as_sym_id(proto_name_v)
        } else {
            panic!("defprotocol: name must be a symbol");
        };
        let mut after_name = v::rest(rest);
        let maybe_doc = v::first(after_name);
        if crate::collections::is_string(maybe_doc) {
            after_name = v::rest(after_name);
        }
        // (The protocol's name is bound to its symbol value by the
        // top-level thunk emitted at the bottom of this function.)

        // Iterate method specs. Each spec must be `(method-name [args...])`
        // (or similar list form). Silently skipping malformed specs would
        // produce a working-but-incomplete protocol with no error — a
        // typo in defprotocol becomes a missing wrapper that surfaces
        // far away as "method not found".
        for spec in v::list_iter(after_name) {
            if !v::is_ptr(spec) || !crate::collections::is_list(spec) {
                panic!(
                    "defprotocol: each method spec must be a list \
                     starting with the method name, got {:#018x}",
                    spec
                );
            }
            let method_name_v = v::first(spec);
            if !v::is_sym_id(method_name_v) {
                panic!(
                    "defprotocol: method spec head must be a symbol \
                     (the method name), got {:#018x}",
                    method_name_v
                );
            }
            let method_sym = v::as_sym_id(method_name_v);
            self.emit_protocol_method_wrapper(method_sym);
        }
        // Top-level thunk: bind `ProtoName` Var → 'ProtoName so
        // `(satisfies? ProtoName x)` resolves.
        *self.anon_counter += 1;
        let thunk_name = format!("__defprotocol_bind_{}", *self.anon_counter);
        let thunk_fref = self.dyn_module.module_builder.declare_func(&thunk_name, &[], Some(Type::I64));
        {
            let mut fb = self.dyn_module.module_builder.define_func(thunk_fref);
            let proto_const = fb.iconst(Type::I64, v::encode_sym_id(proto_sym_id) as i64);
            fb.safepoint(&[]);
            fb.call(self.externs.def_value, &[proto_const, proto_const]);
            let nil_v = fb.iconst(Type::I64, v::NIL as i64);
            fb.ret(nil_v);
            self.dyn_module.module_builder.finish_func(thunk_fref, fb);
        }
        TopResult::Expr(thunk_fref)
    }

    /// Generate a top-level fn `name` that dispatches via
    /// `__method_lookup` and calls the receiver's method impl.
    fn emit_protocol_method_wrapper(&mut self, method_sym: u32) {
        let method_name = self.sym.name(method_sym).to_string();
        // Skip if already defined (a previous defprotocol may have
        // declared the same method, or a method with the same
        // source-level name as a builtin extern).
        if self.func_refs.contains_key(&method_name) {
            return;
        }
        let fref = self
            .dyn_module
            .module_builder
            .declare_func(&method_name, &[Type::I64, Type::I64], Some(Type::I64));
        self.func_refs
            .insert(method_name, FnEntry::DefFn { fref });

        let mut fb = self.dyn_module.module_builder.define_func(fref);
        let entry = fb.entry_block();
        let _self_fn = fb.block_param(entry, 0);
        let args_list = fb.block_param(entry, 1);

        // receiver = first(args_list)
        fb.safepoint(&[args_list]);
        let receiver = fb
            .call(self.externs.reader_list_first, &[args_list])
            .expect("first returns a value");

        // fn_obj = __method_lookup('method, receiver)
        let method_const = fb.iconst(Type::I64, v::encode_sym_id(method_sym) as i64);
        fb.safepoint(&[args_list, receiver]);
        let fn_obj = fb
            .call(self.externs.method_lookup, &[method_const, receiver])
            .expect("__method_lookup returns a value");

        // Trampoline through Fn.func_ref → call_table → indirect call.
        fb.safepoint(&[fn_obj, args_list]);
        let result = self.lower_indirect_call(&mut fb, fn_obj, &[fn_obj, args_list]);
        fb.ret(result);
        self.dyn_module.module_builder.finish_func(fref, fb);
    }

    /// `(extend-type T (Proto (m1 [this …] body) (m2 …) …) (Proto2 …))`
    ///
    /// For each method, compile its body as an anonymous fn-expr
    /// (so it has the closure ABI: `(self_fn, args_list)`) and emit
    /// a top-level thunk that allocates the Fn obj and registers
    /// it via `__register_method(method-sym, type-sym, fn)`.
    fn intern_dot_field(&mut self, fid: u32) -> u32 {
        let s = format!(".-{}", self.sym.name(fid));
        self.sym.intern(&s)
    }

    /// Build a single body form `((let [f0 (.-f0 self) f1 (.-f1 self) …] body…))`
    /// — a one-element list containing a `let` whose body is the
    /// original body forms. Returned in the same shape `compile_extend_type`
    /// already passes around (a list of body forms), so the caller's
    /// `Clause { body: … }` plumbing keeps working.
    fn wrap_body_with_field_bindings(
        &mut self,
        self_id: u32,
        fields: &[u32],
        body_forms: u64,
    ) -> u64 {
        let let_sym = v::encode_sym_id(self.sym.intern("let"));
        let do_sym = v::encode_sym_id(self.sym.intern("do"));
        let self_sym_value = v::encode_sym_id(self_id);

        // Build the binding vector: [f0 (.-f0 self) f1 (.-f1 self) …].
        // We use a vector literal (the natural shape for `let` bindings)
        // built via alloc_vector — the elements are field-name and
        // dot-form pairs.
        // Pre-intern all dot-field symbols outside the GC scope (so the
        // `&mut self.sym` borrow doesn't fight the closure capture).
        let dot_syms: Vec<u32> = fields
            .iter()
            .map(|&fid| self.intern_dot_field(fid))
            .collect();
        dynobj::roots::with_scope(64, |scope| {
            let mut binding_items: Vec<u64> = Vec::with_capacity(fields.len() * 2);
            for (i, &fid) in fields.iter().enumerate() {
                let field_sym_val = v::encode_sym_id(fid);
                let dot_field_sym = v::encode_sym_id(dot_syms[i]);
                let inner_tail = v::alloc_list_cell_from_raw(
                    scope,
                    self_sym_value,
                    v::NIL,
                );
                let dot_call = v::alloc_list_cell_from_raw(
                    scope,
                    dot_field_sym,
                    inner_tail.get(),
                );
                binding_items.push(field_sym_val);
                binding_items.push(dot_call.get());
            }
            let bindings_vec =
                crate::collections::alloc_vector(scope, &binding_items).get();

            // Body of the let: a `do` block over the original body
            // forms. Using `do` keeps multiple body forms working
            // without rebuilding the let body shape.
            let do_form = v::alloc_list_cell_from_raw(scope, do_sym, body_forms);

            // (let [bindings] (do body…))
            let let_tail =
                v::alloc_list_cell_from_raw(scope, do_form.get(), v::NIL);
            let let_after_bindings = v::alloc_list_cell_from_raw(
                scope,
                bindings_vec,
                let_tail.get(),
            );
            let let_form = v::alloc_list_cell_from_raw(
                scope,
                let_sym,
                let_after_bindings.get(),
            );

            // Wrap as a single-element body forms list.
            v::alloc_list_cell_from_raw(scope, let_form.get(), v::NIL).get()
        })
    }

    fn compile_extend_type(&mut self, form: u64) -> TopResult {
        let rest = v::rest(form);
        let type_name_v = v::first(rest);
        if !v::is_sym_id(type_name_v) {
            panic!("extend-type: type name must be a symbol");
        }
        let type_sym = v::as_sym_id(type_name_v);

        // Parse: (extend-type T Proto (m [this …] body) Proto2 (m2 …))
        // — protocol heads are interleaved with method-impl lists.
        // We only need the method impls; protocol names are
        // currently informational (until defprotocol grows real
        // semantics).
        let mut method_impls: Vec<u64> = Vec::new();
        // Walk the body: Symbol = protocol marker (start of a new
        // protocol's method-impl block); List = method impl. We collect
        // BOTH the protocol markers (so satisfies? can answer for each
        // declared protocol — including marker protocols with no
        // methods) and the impls.
        let mut protocol_syms: Vec<u32> = Vec::new();
        for x in v::list_iter(v::rest(rest)) {
            if v::is_sym_id(x) {
                protocol_syms.push(v::as_sym_id(x));
                continue;
            }
            method_impls.push(x);
        }

        // Generate a registration thunk that compiles each method,
        // wraps it in an Fn, and calls __register_method.
        *self.anon_counter += 1;
        let reg_name = format!("__extend_type_{}", *self.anon_counter);
        let reg_fref = self.dyn_module.module_builder.declare_func(&reg_name, &[], Some(Type::I64));

        // Group impls by method name first — protocols can declare
        // multiple arities of the same method, and we need them all
        // to dispatch through one Fn (otherwise the last-registered
        // arity silently overwrites the others). Order is preserved
        // per group; cross-group order is the order method-names
        // first appear.
        let mut method_order: Vec<u32> = Vec::new();
        let mut method_clauses: HashMap<u32, Vec<Clause>> = HashMap::new();
        for impl_form in &method_impls {
            let method_name_v = v::first(*impl_form);
            if !v::is_sym_id(method_name_v) {
                panic!("extend-type: method name must be a symbol");
            }
            let method_sym = v::as_sym_id(method_name_v);
            let after_name = v::rest(*impl_form);
            let arg_vec = v::first(after_name);
            let body_forms = v::rest(after_name);
            let params = parse_params(arg_vec, self.sym);

            // Field auto-binding (Clojure-style): when extending a
            // user deftype, wrap the body in
            //   (let [field0 (.-field0 self) …] body…)
            // so method bodies can refer to fields by bare name.
            let body_with_fields = if let Some(fields) =
                self.deftype_fields.get(&type_sym).cloned()
                && !params.fixed.is_empty()
            {
                let self_id = params.fixed[0];
                self.wrap_body_with_field_bindings(self_id, &fields, body_forms)
            } else {
                body_forms
            };

            if !method_clauses.contains_key(&method_sym) {
                method_order.push(method_sym);
            }
            method_clauses
                .entry(method_sym)
                .or_default()
                .push(Clause { params, body: body_with_fields });
        }

        // Compile one Fn per method name, multi-arity bodies use the
        // existing dispatch in compile_closure_body_clauses.
        let mut compiled_methods: Vec<(u32, FuncRef)> = Vec::new();
        for method_sym in &method_order {
            let clauses = method_clauses.remove(method_sym).unwrap();
            *self.anon_counter += 1;
            let method_func_name = format!(
                "__method_{}_{}_{}",
                self.sym.name(type_sym),
                self.sym.name(*method_sym),
                *self.anon_counter
            );
            let method_fref = self.dyn_module.module_builder.declare_func(
                &method_func_name,
                &[Type::I64, Type::I64],
                Some(Type::I64),
            );
            self.compile_closure_body_clauses(method_fref, &clauses, &[]);
            compiled_methods.push((*method_sym, method_fref));
        }

        // Build the registration thunk.
        {
            let mut fb = self.dyn_module.module_builder.define_func(reg_fref);
            for (method_sym, method_fref) in &compiled_methods {
                // Allocate a closure wrapping the method body. No
                // captures (extend-type methods close over nothing —
                // any `self` reference is a regular fn param). Arity
                // word accepts any count; the body's own dispatcher
                // handles arity validation.
                let arity_word = encode_arity(0, true) as i64;
                let fn_val = self.closures.make(
                    &mut fb,
                    MakeClosure {
                        body_ref: *method_fref,
                        arity_word,
                        captures: &[],
                        extras: &[],
                    },
                    &[],
                );
                // Register method.
                let m_sym_const = fb.iconst(Type::I64, v::encode_sym_id(*method_sym) as i64);
                let t_sym_const = fb.iconst(Type::I64, v::encode_sym_id(type_sym) as i64);
                fb.safepoint(&[fn_val]);
                fb.call(self.externs.register_method, &[m_sym_const, t_sym_const, fn_val]);
            }
            // Register protocol-membership for every protocol named in
            // this extend-type, including marker protocols with no
            // methods. Lets `(satisfies? IList xs)` answer true when
            // the type only "claims" IList without implementing
            // anything.
            for proto_sym in &protocol_syms {
                let p_sym_const =
                    fb.iconst(Type::I64, v::encode_sym_id(*proto_sym) as i64);
                let t_sym_const =
                    fb.iconst(Type::I64, v::encode_sym_id(type_sym) as i64);
                fb.safepoint(&[]);
                fb.call(self.externs.register_protocol_member, &[t_sym_const, p_sym_const]);
            }
            let nil_v = fb.iconst(Type::I64, v::NIL as i64);
            fb.ret(nil_v);
            self.dyn_module.module_builder.finish_func(reg_fref, fb);
        }
        TopResult::Expr(reg_fref)
    }

    fn compile_def_like(
        &mut self,
        form: u64,
        kw: &str,
        expect_fn: bool,
    ) -> (String, FuncRef, u64) {
        let rest = v::rest(form);
        let name_v = v::first(rest);
        if !v::is_sym_id(name_v) {
            panic!("{}: name must be a symbol", kw);
        }
        let name = self.sym.name(v::as_sym_id(name_v)).to_string();
        let expr = v::first(v::rest(rest));

        if !expect_fn {
            unreachable!("non-fn def shape not yet supported")
        }
        if !v::is_ptr(expr) {
            panic!(
                "{}: only `({} NAME (fn [args] body))` is supported in this checkpoint; \
                 got {} of {} bound to non-fn",
                kw, kw, kw, name
            );
        }
        let fn_head = v::first(expr);
        if !v::is_sym_id(fn_head) || self.sym.name(v::as_sym_id(fn_head)) != "fn" {
            panic!(
                "{}: only `({} NAME (fn [args] body))` is supported in this checkpoint; \
                 got {} of {} bound to non-fn",
                kw, kw, kw, name
            );
        }
        let fn_rest = v::rest(expr);
        let clauses = parse_clauses(fn_rest, self.sym);
        // Multi-arity defns are marked variadic at the Fn level so the
        // call-site arity_check accepts any `n >= min`; the body's own
        // dispatch then enforces an exact match. Without this, a
        // single-arity-per-clause Fn rejects every call that isn't
        // exactly `min_arity`.
        let multi = clauses.len() > 1;
        let (fref, arity, is_variadic) = self.compile_def_fn(&name, clauses);
        let arity_word = encode_arity(arity, is_variadic || multi);
        (name, fref, arity_word)
    }

    /// Shared backbone for compiling a top-level `def`d function body.
    ///
    /// For non-variadic params `[a b c]`, the FuncRef is declared
    /// with N direct I64 params and the body binds each from the
    /// matching block-param index.
    ///
    /// For variadic params `[a b & rs]`, the FuncRef is declared
    /// with a SINGLE `args_list` param. The body's prologue walks
    /// the list to bind each fixed param via `(first list)` /
    /// `(rest list)`, then binds `rs` to whatever remains. Static
    /// callers pack their args into a list at compile time and
    /// `fb.call` with one argument.
    ///
    /// Returns (FuncRef, arity-info-for-Var-bookkeeping). The arity
    /// reported is the smallest `min_arity` across all clauses; the
    /// heap `Fn` driver stores `encode_arity(min, is_variadic)` in
    /// the Fn's arity field for runtime arity-check purposes
    /// (variadic flag is set if ANY clause is variadic).
    fn compile_def_fn(
        &mut self,
        name: &str,
        clauses: Vec<Clause>,
    ) -> (FuncRef, usize, bool) {
        // Unified ABI: every user-defined fn — def-fn or closure —
        // takes `(self_fn, args_list)`. For def-fns there's no
        // closure receiver, but having the same shape as closures
        // means a def-fn can be passed as a value (#18) and called
        // through the same indirect path that calls closures.
        // Static call sites pass NIL as `self_fn`.
        let fref = self
            .dyn_module
            .module_builder
            .declare_func(name, &[Type::I64, Type::I64], Some(Type::I64));
        self.func_refs
            .insert(name.to_string(), FnEntry::DefFn { fref });

        let min_arity = clauses
            .iter()
            .map(|c| c.params.min_arity())
            .min()
            .unwrap_or(0);
        let is_variadic = clauses.iter().any(|c| c.params.is_variadic());

        self.compile_clauses_body(fref, &clauses, /*has_self_fn=*/ false);
        (fref, min_arity, is_variadic)
    }

    /// Lower a function whose body is a list of forms `(e1 e2 ... eN)`.
    /// Forms run in sequence; the last produces the return value. An
    /// empty body returns nil.
    ///
    /// The function's entry block immediately jumps to a fresh
    /// `loop_header` block whose params mirror the user's params.
    /// `recur` targets `loop_header`, not `entry`. This indirection
    /// keeps `entry` predecessor-free (the regalloc treats entry
    /// specially because the calling convention pre-loads its params
    /// into specific registers; back-edges into entry would require
    /// re-establishing that ABI on every iteration).
    /// Compile a def-fn body. Same `(self_fn, args_list)` ABI as a
    /// closure body — the only difference is that def-fns have no
    /// captures, so we pass an empty capture list to the kit. `recur`
    /// inside a def-fn body re-enters the kit's loop_header with
    /// `(self_fn, args_list_repacked)` just like a closure recur;
    /// self_fn passes through unused but maintains shape.
    fn compile_clauses_body(
        &mut self,
        fref: FuncRef,
        clauses: &[Clause],
        has_self_fn: bool,
    ) {
        debug_assert!(!has_self_fn, "def-fn path");
        // The closure-body path handles 0 captures correctly — same IR,
        // no capture-load loop. Reuse it verbatim.
        self.compile_closure_body_clauses(fref, clauses, &[]);
    }

    /// Pack `args.len()` already-evaluated arg Values into a Clojure
    /// list at the call site. Returns the list-head IR Value.
    ///
    /// The list is built right-to-left via repeated `cons` calls.
    /// Each `cons` may allocate, so each iteration emits a safepoint
    /// whose live set includes:
    ///   - the current `env` bindings,
    ///   - the in-progress accumulator `acc`,
    ///   - **every arg that hasn't been consed yet** (i.e. args[..i]
    ///     when we're about to cons args[i]). Without these, a
    ///     relocating GC during the cons would invalidate the args
    ///     we're about to use in earlier iterations.
    fn build_args_list(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        args: &[Value],
    ) -> Value {
        // `__cons` is a private alias of the raw cons extern — `cons`
        // itself gets shadowed by core.clj's seq-aware `(defn cons …)`,
        // which would turn this static call into a malformed indirect
        // call (def-fn ABI expects (self_fn, args_list), not raw args).
        let mut acc = fb.iconst(Type::I64, v::NIL as i64);
        for (i, &a) in args.iter().enumerate().rev() {
            let mut live = env.live_values();
            live.push(acc);
            live.push(a);
            // Future iterations will cons args[..i]. Keep them live
            // across this allocation.
            live.extend_from_slice(&args[..i]);
            fb.safepoint(&live);
            acc = fb
                .call(self.externs.cons, &[a, acc])
                .expect("cons returns a value");
        }
        acc
    }

    /// Emit IR that raises the given `exc_value` as a JIT exception.
    /// Calls `__raise_exception` through the control-aware indirect
    /// path so the outcome propagates up. The current block is
    /// terminated; the caller must switch to a fresh block to
    /// continue emission.
    fn emit_raise(&mut self, fb: &mut FunctionBuilder, exc_value: Value) {
        let fr_value = fb.iconst(Type::I64, self.externs.raise_exception.as_u32() as i64);
        fb.safepoint(&[exc_value]);
        fb.call_via_func_ref(self.call_table_base, fr_value, &[exc_value], Some(Type::I64));
        fb.unreachable();
    }

    /// After calling a throw-on-failure extern (one that returns
    /// `NIL` on success or an exception record on failure), emit:
    ///   if result != NIL { raise result }
    ///   else { continue }
    /// The current block is left as the "continue" path.
    fn emit_throw_if_not_nil(&mut self, fb: &mut FunctionBuilder, result: Value) {
        let nil_const = fb.iconst(Type::I64, v::NIL as i64);
        let is_nil = fb.icmp(CmpOp::Eq, result, nil_const);
        let continue_bb = fb.create_block(&[]);
        let throw_bb = fb.create_block(&[]);
        fb.br_if(is_nil, continue_bb, &[], throw_bb, &[]);
        fb.switch_to_block(throw_bb);
        self.emit_raise(fb, result);
        // Open a dead block in case the lowerer wants to emit
        // continuation in the throw_bb (it shouldn't — emit_raise
        // terminates).
        fb.switch_to_block(continue_bb);
    }

    /// Emit an indirect call through an `Fn` heap object via
    /// [`ClosureKit::call`]. Uniform closure ABI `(self_fn, args_list)`.
    ///
    /// `args` is the user-side arg slice the call site is forwarding,
    /// in the kit's expected shape: for `ArgsList`, that's a single
    /// pre-packed list value (the kit prepends `fn_obj` as `self_fn`
    /// itself). Callers that have a `(self_fn, args_list)` pair on
    /// hand should pass just `&[args_list]` — passing both reproduces
    /// the legacy "prepend twice" bug, so this method strips the
    /// duplicate when present.
    fn lower_indirect_call(
        &self,
        fb: &mut FunctionBuilder,
        fn_obj: Value,
        args: &[Value],
    ) -> Value {
        // Legacy call sites passed `&[fn_obj, args_list]`; the kit
        // already adds `fn_obj` as the self_fn slot, so accept either
        // shape and forward just the user args.
        let user_args: &[Value] = if args.first().copied() == Some(fn_obj) {
            &args[1..]
        } else {
            args
        };
        // Caller-side safepoint contract preserved: the live set
        // varies by call site, so callers emit their own safepoint
        // before this. We pass an empty roots vec so the kit's
        // safepoint matches what was here before.
        self.closures
            .call(fb, self.call_table_base, fn_obj, user_args, &[])
    }

    /// Static-dispatch a call site whose head resolves to a known
    /// `FnEntry`. Externs and non-variadic user fns get a direct
    /// `fb.call`; variadic user fns get their args packed into a
    /// list first.
    fn lower_static_call(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        entry: FnEntry,
        arg_forms: &[u64],
    ) -> Value {
        let n = arg_forms.len();
        match entry {
            FnEntry::Extern { fref, arity } => {
                if n != arity {
                    let mut name = String::from("?");
                    for (k, v) in self.func_refs.iter() {
                        if let FnEntry::Extern { fref: f, .. } = v {
                            if *f == fref {
                                name = k.clone();
                                break;
                            }
                        }
                    }
                    panic!(
                        "extern arity mismatch: `{}` expects {}, got {}",
                        name, arity, n
                    );
                }
                let arg_vals: Vec<Value> = arg_forms
                    .iter()
                    .map(|a| self.lower_expr(fb, env, *a))
                    .collect();
                let mut live = env.live_values();
                live.extend_from_slice(&arg_vals);
                fb.safepoint(&live);
                fb.call(fref, &arg_vals)
                    .expect("extern returns a value")
            }
            FnEntry::DefFn { fref } => {
                // Unified ABI: def-fn takes `(self_fn, args_list)`.
                // Pack user args into a list, pass NIL for self_fn.
                let arg_vals: Vec<Value> = arg_forms
                    .iter()
                    .map(|a| self.lower_expr(fb, env, *a))
                    .collect();
                let list = self.build_args_list(fb, env, &arg_vals);
                let nil_self = fb.iconst(Type::I64, v::NIL as i64);
                let mut live = env.live_values();
                live.push(list);
                fb.safepoint(&live);
                fb.call(fref, &[nil_self, list])
                    .expect("def-fn returns a value")
            }
        }
    }

    // (`compile_function_body` removed — replaced by the unified
    // single-list ABI handled by `compile_variadic_body`.)

    fn lower_expr(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        // Clear the non-returning flag — only the most-recent
        // expression's exit shape should be visible to its caller.
        // (Set again by `lower_throw` / `lower_recur` if applicable.)
        self.last_expr_non_returning = false;
        // Self-evaluating immediates.
        if v::is_number(form) || v::is_nil(form) || form == v::TRUE || form == v::FALSE {
            return fb.iconst(Type::I64, form as i64);
        }
        // Bare symbol resolution path. In order:
        //   1. Lexical env (let/loop bindings, fn params, captures).
        //   2. A `def`d top-level Var in `clojure.core`. We pin the
        //      Var pointer in the literal pool and emit a load of
        //      `Var.root`, which gives the user the CURRENT `Fn` heap
        //      object — re-evaluated each time so `(def-redef)` is
        //      visible to subsequent reads.
        //   3. Otherwise: undefined.
        if v::is_sym_id(form) {
            let id = v::as_sym_id(form);
            if let Some(val) = env.lookup(id) {
                return val;
            }
            // Try to resolve as a Var in clojure.core at compile
            // time (fast path — pin the Var ptr in the literal pool
            // and emit a single load of `Var.root`).
            let sym_value = v::encode_sym_id(id);
            let var_ptr = crate::namespace::ns_lookup(self.core_ns, sym_value);
            if v::is_ptr(var_ptr) {
                let lit_idx = self.literal_pool.push(var_ptr) as u32;
                let var_loaded = fb.gc_literal(LiteralRef::from_u32(lit_idx));
                let var_obj = fb.payload(var_loaded);
                let var_obj_i64 = fb.bitcast(var_obj, Type::I64);
                return fb.load(Type::I64, var_obj_i64, crate::namespace::var_root_offset());
            }
            // No Var, no func_refs entry, not bound locally. This is
            // an undefined symbol. Match Clojure's behavior — error at
            // compile time rather than emit a runtime lookup. Forward
            // references that need to resolve later require a
            // `(declare …)` ahead of use, which interns an unbound Var
            // so this branch never fires.
            //
            // Why not silently emit a runtime lookup? Because then
            // forward-referenced MACROS silently degrade to function
            // calls — args get eagerly evaluated and bypass the macro's
            // structural rewrite. We caught this only when `count`
            // (compiled before `cond`) panicked deep inside one of
            // `cond`'s arms.
            panic!(
                "Unable to resolve symbol: {} in this context",
                self.sym.name(id)
            );
        }
        if !v::is_ptr(form) {
            panic!("unrecognized form: 0x{:016x}", form);
        }
        // Self-evaluating compound literals (string / keyword / vector
        // / map / set). The expander has already normalized any macro
        // result that came back as a non-list seqable (PList, Cons, …)
        // to a built-in __ReaderList, so we only distinguish "list"
        // from "everything else."
        //
        // Vectors and maps are tricky: in Clojure, `[a b c]` is NOT a
        // literal vector — it evaluates each element. Only if ALL
        // elements are themselves self-evaluating (numbers, keywords,
        // nested all-literal vectors, …) can we pin to the literal
        // pool. Otherwise, lower each element and build at runtime.
        if !crate::collections::is_list(form) {
            if crate::collections::is_vector(form)
                && vector_needs_runtime_eval(form)
            {
                return self.lower_vector_literal(fb, env, form);
            }
            // TODO: same treatment for maps with live elements; today
            // {:k (foo)} is pinned as-is. Tracked in TODO.md.
            let idx = self.literal_pool.push(form) as u32;
            return fb.gc_literal(LiteralRef::from_u32(idx));
        }
        // List form: head dispatches on special form vs. call.
        let head = v::first(form);
        if v::is_sym_id(head) {
            let head_name = self.sym.name(v::as_sym_id(head)).to_string();
            match head_name.as_str() {
                "if" => return self.lower_if(fb, env, v::rest(form)),
                "do" => return self.lower_do(fb, env, v::rest(form)),
                "let" => return self.lower_let(fb, env, v::rest(form)),
                "loop" => return self.lower_loop(fb, env, v::rest(form)),
                "recur" => return self.lower_recur(fb, env, v::rest(form)),
                "quote" => return self.lower_quote(fb, v::rest(form)),
                "def" => panic!("def is only valid at top level"),
                "fn" => return self.lower_fn_expr(fb, env, v::rest(form)),
                "set!" => return self.lower_set_bang(fb, env, v::rest(form)),
                "throw" => return self.lower_throw(fb, env, v::rest(form)),
                "try" => return self.lower_try(fb, env, v::rest(form)),
                _ => {}
            }
        }
        self.lower_call(fb, env, form)
    }

    /// `(set! (.-field receiver) val)` — store `val` into the named
    /// field of `receiver`. Receiver must be a record; field is a
    /// `(.-name receiver)` form whose head we destructure to get the
    /// field name. Returns the stored value (matches Clojure's
    /// expression semantics).
    fn lower_set_bang(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        rest: u64,
    ) -> Value {
        let place = v::first(rest);
        let val_form = v::first(v::rest(rest));
        if !v::is_ptr(place) || !crate::collections::is_list(place) {
            panic!("set!: target must be a (.-field receiver) form");
        }
        let head = v::first(place);
        if !v::is_sym_id(head) {
            panic!("set!: target must be a (.-field receiver) form");
        }
        let head_name = self.sym.name(v::as_sym_id(head)).to_string();
        let field_name = head_name
            .strip_prefix(".-")
            .unwrap_or_else(|| panic!("set!: target must be a (.-field receiver) form"));
        let receiver_form = v::first(v::rest(place));
        let receiver = self.lower_expr(fb, env, receiver_form);
        let val = self.lower_expr(fb, env, val_form);

        let field_sym = self.sym.intern(field_name);
        let field_const = fb.iconst(Type::I64, v::encode_sym_id(field_sym) as i64);
        let mut live = env.live_values();
        live.push(receiver);
        live.push(val);
        fb.safepoint(&live);
        fb.call(self.externs.record_set_field, &[receiver, field_const, val])
            .expect("__record_set_field returns a value")
    }

    /// `(throw v)` — uniform path: call `__raise_exception` through
    /// the JIT call table (indirect, control-aware). The stub
    /// returns `JitOutcome::Exception(v)`. Plain Call sites on the
    /// way up auto-propagate the outcome; the nearest enclosing
    /// `Invoke` (emitted by `lower_try`) catches it and routes to
    /// the handler block. If no Invoke is ever reached, `eval`
    /// surfaces the uncaught exception as a panic.
    ///
    /// Same mechanism for in-function and cross-function throws.
    /// No need for separate CFG-jump paths.
    fn lower_throw(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let val_form = v::first(args);
        let val = self.lower_expr(fb, env, val_form);
        let fr_value = fb.iconst(Type::I64, self.externs.raise_exception.as_u32() as i64);
        let mut live = env.live_values();
        live.push(val);
        fb.safepoint(&live);
        fb.call_via_func_ref(self.call_table_base, fr_value, &[val], Some(Type::I64));
        fb.unreachable();
        let dead = fb.create_block(&[]);
        fb.switch_to_block(dead);
        self.last_expr_non_returning = true;
        fb.iconst(Type::I64, v::NIL as i64)
    }

    /// `(try body... (catch T1 e h1)... (catch Tn e hn) (finally f...))`
    ///
    /// Multi-arm catches with type filtering, plus an optional
    /// trailing `(finally ...)`.
    ///
    /// ## Mechanism without `finally`
    ///
    ///  - Body lowers as a synthesized internal IR fn taking the
    ///    captured locals as args.
    ///  - Outer fn `fb.invoke`s it with normal=normal_bb and
    ///    exception=dispatch_bb.
    ///  - Throws inside body produce `JitOutcome::Exception(v)`.
    ///    The Invoke catches and routes the thrown value to
    ///    dispatch_bb's first param.
    ///  - `dispatch_bb` walks the catch arms in order, comparing
    ///    via `__exception_type_matches`. First match wins.
    ///    `_` / `:default` are catch-all. No-match → re-throw.
    ///
    /// ## Mechanism with `finally`
    ///
    /// Naïvely running finally after the dispatch (in the outer fb)
    /// misses the case where a **catch handler itself throws**: that
    /// throw flies past finally up to the outer caller. To make
    /// finally run on every exit (including handler-throws), the
    /// try-without-finally is itself synthesized as a wrapper fn,
    /// and the outer's invoke catches anything it propagates:
    ///
    /// ```text
    ///   outer:
    ///     invoke __try_wrapper_<n>(caps)
    ///         normal    -> wrapper_normal_bb([result])
    ///         exception -> wrapper_exception_bb([thrown])
    ///
    ///   wrapper_normal_bb:
    ///     lower(finally_body)
    ///     jump merge_bb([result])
    ///
    ///   wrapper_exception_bb:
    ///     lower(finally_body)
    ///     call_via_func_ref __raise_exception(thrown)
    ///     unreachable
    ///
    ///   __try_wrapper_<n>:                     ; synthesized
    ///     // full try-without-finally lowering inside.
    ///     // body throws or arm throws => wrapper exits with
    ///     // Exception outcome => outer catches => finally runs.
    /// ```
    ///
    /// This way the finally body's IR is emitted exactly twice (the
    /// outer's normal and exception paths), not once per arm.
    ///
    /// ## Type-filter resolution
    ///
    /// A catch's `T` is a *symbol* in the source. At compile time
    /// we intern it; at runtime the type check compares the thrown
    /// record's `type_name` (sym-id) against the interned id. There
    /// is no class hierarchy — exact match only.
    fn lower_try(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let forms: Vec<u64> = v::list_iter(args).collect();
        // Split forms into body / catch arms / optional finally.
        let mut body_forms: Vec<u64> = Vec::new();
        let mut catch_arms: Vec<CatchArm> = Vec::new();
        let mut finally_body: Option<u64> = None;
        let mut seen_catch_or_finally = false;
        for &form in &forms {
            if Self::is_finally_form(self.sym, form) {
                if finally_body.is_some() {
                    panic!("try: at most one (finally ...) clause allowed");
                }
                finally_body = Some(v::rest(form));
                seen_catch_or_finally = true;
            } else if Self::is_catch_form(self.sym, form) {
                if finally_body.is_some() {
                    panic!("try: catch clauses must come before (finally ...)");
                }
                catch_arms.push(Self::parse_catch_arm(self.sym, form));
                seen_catch_or_finally = true;
            } else {
                if seen_catch_or_finally {
                    panic!(
                        "try: body form after a catch/finally is not allowed"
                    );
                }
                body_forms.push(form);
            }
        }

        if catch_arms.is_empty() && finally_body.is_none() {
            panic!("try: must have at least one (catch ...) or (finally ...) clause");
        }

        // Without finally: emit the try-catch directly in `fb`.
        if finally_body.is_none() {
            return self.lower_try_no_finally(fb, env, &body_forms, &catch_arms);
        }

        // With finally: synthesize a wrapper fn that contains the
        // entire try-catch (body fn + dispatch + arm bodies). Outer
        // invokes the wrapper; finally runs on both the normal and
        // exception paths. A throw from a catch handler exits the
        // wrapper with `JitOutcome::Exception`, which the outer's
        // invoke catches → finally → re-raise. This is what makes
        // finally run for handler-thrown exceptions.
        let finally_body = finally_body.unwrap();
        self.lower_try_with_finally(fb, env, &body_forms, &catch_arms, finally_body)
    }

    /// Lower a try-catch (no finally). Synthesizes a body fn via
    /// [`InlineBody`] holding the try-body's free vars as positional
    /// captures, then emits `fb.invoke` into outer `fb` with dispatch
    /// + arms running in the outer frame.
    fn lower_try_no_finally(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        body_forms: &[u64],
        catch_arms: &[CatchArm],
    ) -> Value {
        // Free-var analyze body to find outer captures.
        let mut captures: Vec<u32> = Vec::new();
        for &form in body_forms {
            let raw = {
                let sym_ref = &*self.sym;
                crate::freevars::free_vars_in_form(form, &[], &|id| {
                    sym_ref.name(id).to_string()
                })
            };
            for id in raw {
                if env.lookup(id).is_some() && !captures.contains(&id) {
                    captures.push(id);
                }
            }
        }

        *self.anon_counter += 1;
        let name = format!("__try_body_{}", *self.anon_counter);
        let body = InlineBody::declare(&mut self.dyn_module.module_builder, &name, captures.len(), 0);

        // Open + lower the body. The kit doesn't hold a borrow on
        // `&mut self.dyn_module.module_builder` between `open` and `finish`, so recursive
        // `self.lower_expr` works normally here.
        {
            let (mut inner_fb, cap_vals, _extras) = body.open(&mut self.dyn_module.module_builder);
            let mut inner_env = Env::new();
            for (i, &cap_id) in captures.iter().enumerate() {
                inner_env.bind(cap_id, cap_vals[i]);
            }
            let body_value = if body_forms.is_empty() {
                inner_fb.iconst(Type::I64, v::NIL as i64)
            } else {
                let mut last = inner_fb.iconst(Type::I64, v::NIL as i64);
                for form in body_forms {
                    last = self.lower_expr(&mut inner_fb, &mut inner_env, *form);
                }
                last
            };
            if !inner_fb.current_block_is_terminated() {
                inner_fb.ret(body_value);
            } else {
                inner_fb.unreachable();
            }
            body.finish(&mut self.dyn_module.module_builder, inner_fb);
        }

        // Invoke from outer `fb`; dispatch + arms emitted here.
        let dispatch_bb = fb.create_block(&[Type::I64]); // thrown
        let normal_bb = fb.create_block(&[Type::I64]);   // body return
        let merge_bb = fb.create_block(&[Type::I64]);    // unified result

        let cap_values: Vec<Value> = captures
            .iter()
            .map(|&id| env.lookup(id).expect("capture missing in env"))
            .collect();
        let mut live = env.live_values();
        live.extend_from_slice(&cap_values);
        body.invoke(fb, &cap_values, &[], normal_bb, dispatch_bb, &live);

        fb.switch_to_block(normal_bb);
        let body_v = fb.block_param(normal_bb, 0);
        fb.jump(merge_bb, &[body_v]);

        fb.switch_to_block(dispatch_bb);
        let thrown = fb.block_param(dispatch_bb, 0);
        self.emit_catch_dispatch(fb, env, thrown, catch_arms, merge_bb);

        fb.switch_to_block(merge_bb);
        fb.block_param(merge_bb, 0)
    }

    /// Lower `(try body... catches... (finally f))` by wrapping the
    /// try-catch in a synthesized fn. The outer fb invokes that
    /// wrapper; finally runs on both normal and exception paths.
    /// This is what makes finally run for handler-thrown exceptions
    /// (the wrapper's call propagates them, outer catches via
    /// invoke's exception block).
    fn lower_try_with_finally(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        body_forms: &[u64],
        catch_arms: &[CatchArm],
        finally_body: u64,
    ) -> Value {
        // Compute total captures: free vars across body + each arm's
        // body (minus that arm's bind name).
        let mut captures: Vec<u32> = Vec::new();
        let push_cap = |captures: &mut Vec<u32>, id: u32| {
            if !captures.contains(&id) {
                captures.push(id);
            }
        };
        for &form in body_forms {
            let raw = {
                let sym_ref = &*self.sym;
                crate::freevars::free_vars_in_form(form, &[], &|id| {
                    sym_ref.name(id).to_string()
                })
            };
            for id in raw {
                if env.lookup(id).is_some() {
                    push_cap(&mut captures, id);
                }
            }
        }
        for arm in catch_arms {
            // The arm's body sees `bind_sym` as a local; everything
            // else free in arm.body that's also bound in the outer
            // env must be captured.
            for elem in v::list_iter(arm.body) {
                let raw = {
                    let sym_ref = &*self.sym;
                    crate::freevars::free_vars_in_form(
                        elem,
                        &[arm.bind_sym],
                        &|id| sym_ref.name(id).to_string(),
                    )
                };
                for id in raw {
                    if env.lookup(id).is_some() {
                        push_cap(&mut captures, id);
                    }
                }
            }
        }

        *self.anon_counter += 1;
        let name = format!("__try_wrapper_{}", *self.anon_counter);
        let wrapper = InlineBody::declare(&mut self.dyn_module.module_builder, &name, captures.len(), 0);

        // Compile wrapper: contains the try-catch (no finally), which
        // itself synthesizes another inline body via `lower_try_no_finally`.
        {
            let (mut wrapper_fb, cap_vals, _extras) = wrapper.open(&mut self.dyn_module.module_builder);
            let mut wrapper_env = Env::new();
            for (i, &cap_id) in captures.iter().enumerate() {
                wrapper_env.bind(cap_id, cap_vals[i]);
            }
            let inner_result = self.lower_try_no_finally(
                &mut wrapper_fb,
                &mut wrapper_env,
                body_forms,
                catch_arms,
            );
            if !wrapper_fb.current_block_is_terminated() {
                wrapper_fb.ret(inner_result);
            } else {
                wrapper_fb.unreachable();
            }
            wrapper.finish(&mut self.dyn_module.module_builder, wrapper_fb);
        }

        // Outer: invoke the wrapper, with finally on both paths.
        let cap_values: Vec<Value> = captures
            .iter()
            .map(|&id| env.lookup(id).expect("capture missing in env"))
            .collect();

        let normal_bb = fb.create_block(&[Type::I64]);    // wrapper result
        let exception_bb = fb.create_block(&[Type::I64]); // thrown
        let merge_bb = fb.create_block(&[Type::I64]);     // unified result

        let mut live = env.live_values();
        live.extend_from_slice(&cap_values);
        wrapper.invoke(fb, &cap_values, &[], normal_bb, exception_bb, &live);

        // Normal path: run finally for side-effects, forward result.
        fb.switch_to_block(normal_bb);
        let result_v = fb.block_param(normal_bb, 0);
        if !v::is_nil(finally_body) {
            env.push();
            let _ = self.lower_do(fb, env, finally_body);
            env.pop();
        }
        if !fb.current_block_is_terminated() {
            fb.jump(merge_bb, &[result_v]);
        }

        // Exception path: run finally for side-effects, re-raise.
        // If finally itself throws, its raise propagates out of this
        // function (replacing the original exception — correct
        // Clojure semantics).
        fb.switch_to_block(exception_bb);
        let thrown = fb.block_param(exception_bb, 0);
        if !v::is_nil(finally_body) {
            env.push();
            let _ = self.lower_do(fb, env, finally_body);
            env.pop();
        }
        if !fb.current_block_is_terminated() {
            self.emit_raise(fb, thrown);
        }

        fb.switch_to_block(merge_bb);
        fb.block_param(merge_bb, 0)
    }

    /// Emit the catch-arm chain starting from the current block (the
    /// dispatch entry). For each arm: if the thrown value matches the
    /// arm's type, bind the name and run the handler (jump to
    /// merge). Else fall through to the next arm. If no arm matches:
    /// re-raise the same value.
    ///
    /// **No finally handling here.** When a `(finally ...)` is
    /// present, the entire try-catch (including this dispatch)
    /// runs inside a wrapper fn so that handler-thrown exceptions
    /// propagate out and the outer's invoke catches them — that's
    /// where the finally body runs. See `lower_try_with_finally`.
    ///
    /// Builds the chain forward. The first arm's test goes in the
    /// current (dispatch) block; each subsequent test goes in a
    /// fresh block that the previous arm's miss-branch targets. The
    /// final miss target is `no_match_bb`, which re-raises.
    fn emit_catch_dispatch(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        thrown: Value,
        arms: &[CatchArm],
        merge_bb: BlockId,
    ) {
        let type_match_fref = self.externs.exception_type_matches;

        // Pre-create per-arm body blocks + the no-match terminal.
        let arm_blocks: Vec<BlockId> =
            (0..arms.len()).map(|_| fb.create_block(&[])).collect();
        let no_match_bb = fb.create_block(&[]);

        // `try` with a `(finally ...)` but no catch arms: the
        // dispatch block has nothing to test — any exception
        // propagates straight to no_match (which runs finally then
        // re-raises).
        if arms.is_empty() {
            fb.jump(no_match_bb, &[]);
        }

        // Walk arms forward, emitting the test in the *current*
        // block. For all but the last arm, the miss-target is a
        // fresh test block that we then switch to. For the last
        // arm, the miss-target is `no_match_bb`.
        for (i, arm) in arms.iter().enumerate() {
            let body_bb = arm_blocks[i];
            let miss_bb = if i + 1 < arms.len() {
                fb.create_block(&[])
            } else {
                no_match_bb
            };

            if let Some(type_sym) = arm.type_sym {
                let expected = fb.iconst(Type::I64, type_sym as i64);
                fb.safepoint(&[thrown]);
                let result = fb
                    .call(type_match_fref, &[thrown, expected])
                    .expect("__exception_type_matches returns a value");
                let true_const = fb.iconst(Type::I64, v::TRUE as i64);
                let cond = fb.icmp(CmpOp::Eq, result, true_const);
                fb.br_if(cond, body_bb, &[], miss_bb, &[]);
            } else {
                // Catch-all: jump unconditionally to the body. Any
                // following arms are unreachable (`miss_bb` will be a
                // dead block) — that's well-formed for the lowerer.
                fb.jump(body_bb, &[]);
            }

            // Advance to the next test block (only if there is one).
            if i + 1 < arms.len() {
                fb.switch_to_block(miss_bb);
            }
        }

        // Emit each arm's handler body.
        for (i, arm) in arms.iter().enumerate() {
            fb.switch_to_block(arm_blocks[i]);
            env.push();
            env.bind(arm.bind_sym, thrown);
            let handler_v = if v::is_nil(arm.body) {
                fb.iconst(Type::I64, v::NIL as i64)
            } else {
                self.lower_do(fb, env, arm.body)
            };
            env.pop();
            if !fb.current_block_is_terminated() {
                fb.jump(merge_bb, &[handler_v]);
            }
        }

        // No-match path: re-raise.
        fb.switch_to_block(no_match_bb);
        self.emit_raise(fb, thrown);
    }

    /// Recognize a `(catch ...)` form.
    fn is_catch_form(sym: &SymbolTable, form: u64) -> bool {
        if !v::is_ptr(form) || !crate::collections::is_list(form) {
            return false;
        }
        let head = v::first(form);
        v::is_sym_id(head) && sym.name(v::as_sym_id(head)) == "catch"
    }

    /// Recognize a `(finally ...)` form.
    fn is_finally_form(sym: &SymbolTable, form: u64) -> bool {
        if !v::is_ptr(form) || !crate::collections::is_list(form) {
            return false;
        }
        let head = v::first(form);
        v::is_sym_id(head) && sym.name(v::as_sym_id(head)) == "finally"
    }

    /// Parse `(catch T name body...)` into a `CatchArm`. `T` is
    /// either a catch-all sentinel (`_` or `:default`) — encoded as
    /// `type_sym = None` — or a symbol naming the exception type.
    fn parse_catch_arm(sym: &SymbolTable, form: u64) -> CatchArm {
        let rest = v::rest(form);
        let type_form = v::first(rest);
        let after_type = v::rest(rest);
        let name_v = v::first(after_type);
        if !v::is_sym_id(name_v) {
            panic!("try/catch: binding name must be a symbol");
        }
        let bind_sym = v::as_sym_id(name_v);
        let body = v::rest(after_type);

        let type_sym = if v::is_sym_id(type_form) {
            let id = v::as_sym_id(type_form);
            let name = sym.name(id);
            if name == "_" {
                None
            } else {
                Some(id)
            }
        } else if v::is_ptr(type_form) && crate::collections::is_keyword(type_form) {
            // `:default` keyword catch-all (matches Clojure ClojureScript
            // convention for non-JVM platforms).
            let kw_sym = crate::collections::keyword_sym_id(type_form);
            let name = sym.name(kw_sym);
            if name == "default" { None } else {
                panic!(
                    "try/catch: keyword type filter must be :default, got :{}",
                    name
                );
            }
        } else {
            panic!(
                "try/catch: type filter must be a symbol or :default, got {:#018x}",
                type_form
            );
        };

        CatchArm {
            type_sym,
            bind_sym,
            body,
        }
    }

    fn lower_if(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        let cond_form = v::first(args);
        let after_cond = v::rest(args);
        let then_form = v::first(after_cond);
        let after_then = v::rest(after_cond);
        let else_form = if v::is_ptr(after_then) {
            v::first(after_then)
        } else {
            v::NIL
        };

        let cond = self.lower_expr(fb, env, cond_form);
        let cond_bool = compute_truthy(fb, cond);

        let then_bb = fb.create_block(&[]);
        let else_bb = fb.create_block(&[]);
        let merge_bb = fb.create_block(&[Type::I64]);
        fb.br_if(cond_bool, then_bb, &[], else_bb, &[]);

        fb.switch_to_block(then_bb);
        let tv = self.lower_expr(fb, env, then_form);
        fb.jump(merge_bb, &[tv]);

        fb.switch_to_block(else_bb);
        let ev = self.lower_expr(fb, env, else_form);
        fb.jump(merge_bb, &[ev]);

        fb.switch_to_block(merge_bb);
        fb.block_param(merge_bb, 0)
    }

    fn lower_do(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        if v::is_nil(args) {
            return fb.iconst(Type::I64, v::NIL as i64);
        }
        let mut last = None;
        for x in v::list_iter(args) {
            last = Some(self.lower_expr(fb, env, x));
        }
        last.unwrap()
    }

    fn lower_let(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // Clojure syntax: (let [name val name val ...] body...)
        // The bindings come in pairs flat in a vector.
        let bindings = v::first(args);
        let body_forms = v::rest(args);
        let pairs: Vec<u64> = crate::collections::seq_iter(bindings).collect();
        if pairs.len() % 2 != 0 {
            panic!("let: bindings must come in name/value pairs");
        }
        env.push();
        for chunk in pairs.chunks(2) {
            let name_v = chunk[0];
            let val_form = chunk[1];
            if !v::is_sym_id(name_v) {
                panic!("let: binding name must be a symbol");
            }
            let val = self.lower_expr(fb, env, val_form);
            env.bind(v::as_sym_id(name_v), val);
        }
        let result = self.lower_do(fb, env, body_forms);
        env.pop();
        result
    }

    /// Lower a vector literal whose elements need runtime
    /// evaluation. Compiles each element via `lower_expr`, builds a
    /// `__ReaderList` of the values right-to-left via `cons`, then
    /// calls `__vector_from_list` to convert.
    fn lower_vector_literal(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        form: u64,
    ) -> Value {
        let elems: Vec<u64> = crate::collections::vector_iter(form).collect();
        let element_vals: Vec<Value> =
            elems.iter().map(|e| self.lower_expr(fb, env, *e)).collect();
        let list = self.build_args_list(fb, env, &element_vals);
        let mut live = env.live_values();
        live.push(list);
        fb.safepoint(&live);
        fb.call(self.externs.vector_from_list, &[list])
            .expect("__vector_from_list returns a value")
    }

    fn lower_quote(&mut self, fb: &mut FunctionBuilder, args: u64) -> Value {
        let datum = v::first(args);
        if v::is_ptr(datum) {
            // Heap datum: pin into the literal pool. The pool is a GC
            // root source, so a moving collection updates the slot in
            // place; emitted code re-reads the (possibly relocated)
            // pointer on every load.
            let idx = self.literal_pool.push(datum) as u32;
            return fb.gc_literal(LiteralRef::from_u32(idx));
        }
        fb.iconst(Type::I64, datum as i64)
    }

    fn lower_call(&mut self, fb: &mut FunctionBuilder, env: &mut Env, form: u64) -> Value {
        let head = v::first(form);
        let arg_forms: Vec<u64> = v::list_iter(v::rest(form)).collect();
        let n = arg_forms.len();

        // Method/field dispatch: head names starting with `.` are
        // sugar for protocol method calls or record field accesses.
        if v::is_sym_id(head) {
            let head_name = self.sym.name(v::as_sym_id(head)).to_string();
            if let Some(field) = head_name.strip_prefix(".-") {
                // (.-field receiver) → record field load.
                if arg_forms.len() != 1 {
                    panic!(".-field: expected exactly one receiver argument");
                }
                let field_sym = self.sym.intern(field);
                let receiver = self.lower_expr(fb, env, arg_forms[0]);
                let field_const = fb.iconst(Type::I64, v::encode_sym_id(field_sym) as i64);
                let mut live = env.live_values();
                live.push(receiver);
                fb.safepoint(&live);
                return fb
                    .call(self.externs.record_get_field, &[receiver, field_const])
                    .expect("__record_get_field returns a value");
            }
            if let Some(method) = head_name.strip_prefix('.') {
                // (.method-name receiver args…) → dispatch.
                if arg_forms.is_empty() {
                    panic!(".method: missing receiver");
                }
                let method_sym = self.sym.intern(method);
                let receiver = self.lower_expr(fb, env, arg_forms[0]);
                let user_args: Vec<Value> = arg_forms[1..]
                    .iter()
                    .map(|a| self.lower_expr(fb, env, *a))
                    .collect();
                let method_const = fb.iconst(Type::I64, v::encode_sym_id(method_sym) as i64);
                let mut live = env.live_values();
                live.push(receiver);
                live.extend_from_slice(&user_args);
                fb.safepoint(&live);
                let fn_obj = fb
                    .call(self.externs.method_lookup, &[method_const, receiver])
                    .expect("__method_lookup returns a value");
                // Pack [receiver, args…] into a single list for the
                // closure-shape call.
                let mut all_args: Vec<Value> = Vec::with_capacity(1 + user_args.len());
                all_args.push(receiver);
                all_args.extend_from_slice(&user_args);
                let args_list = self.build_args_list(fb, env, &all_args);
                let mut live = env.live_values();
                live.push(fn_obj);
                live.push(args_list);
                fb.safepoint(&live);
                return self.lower_indirect_call(fb, fn_obj, &[fn_obj, args_list]);
            }
            // Constructor sugar: head ends with `.` (and isn't bare ".").
            if head_name.ends_with('.') && head_name.len() > 1 {
                // Already handled — `Name.` is registered in
                // func_refs as a regular def-fn by `compile_deftype`.
                // Fall through to the static-call path below.
            }
        }

        // Resolve head: prefer local binding (which is necessarily an
        // Fn value, so dispatch indirectly) over the static func_refs
        // table. Non-symbol heads always go indirect: we evaluate the
        // head expression to a value and dispatch through it.
        if v::is_sym_id(head) {
            let id = v::as_sym_id(head);
            if env.lookup(id).is_none() {
                if let Some(&entry) = self.func_refs.get(&self.sym.name(id)) {
                    return self.lower_static_call(fb, env, entry, &arg_forms);
                }
            }
        }

        // Indirect dispatch.
        //
        // Every closure (variadic or not) follows the uniform single-
        // list ABI: the body's signature is `(self_fn, args_list)`.
        // The call site:
        //   1. Lowers the head expression to a callee value.
        //   2. Lowers each user arg.
        //   3. Calls the runtime arity-check extern, which validates
        //      `n_args` against the callee's encoded arity (handles
        //      both fixed and variadic).
        //   4. Packs the user args into a Clojure list via repeated
        //      `cons` calls.
        //   5. Reads `Fn.func_ref` from the callee and looks up the
        //      code pointer in `call_table[func_ref]`.
        //   6. `call_indirect` the code pointer with `(callee, list)`.
        //
        // The body's prologue then walks the list to bind each fixed
        // param and (if variadic) collects the rest.
        let callee = self.lower_expr(fb, env, head);
        let mut user_args: Vec<Value> = Vec::with_capacity(n);
        for a in &arg_forms {
            user_args.push(self.lower_expr(fb, env, *a));
        }

        let n_const = fb.iconst(Type::I64, n as i64);
        let mut live_for_check = env.live_values();
        live_for_check.push(callee);
        live_for_check.extend_from_slice(&user_args);
        fb.safepoint(&live_for_check);
        // __arity_check returns NIL on success, exception record on
        // failure. emit_throw_if_not_nil routes to __raise_exception
        // on the failure path so (catch ArityException e ...) works.
        let arity_result = fb
            .call(self.externs.arity_check, &[callee, n_const])
            .expect("__arity_check returns a value");
        self.emit_throw_if_not_nil(fb, arity_result);

        // Pack user args into a list.
        let args_list = self.build_args_list(fb, env, &user_args);

        let mut live = env.live_values();
        live.push(callee);
        live.push(args_list);
        fb.safepoint(&live);
        self.lower_indirect_call(fb, callee, &[callee, args_list])
    }

    /// Compile `(fn [args...] body...)` as a value.
    ///
    /// 1. Free-var analyze each clause's body to find captures
    ///    (symbols bound in the enclosing env but not by the fn's
    ///    own params).
    /// 2. Declare the inner body fn via the kit (uniform
    ///    `(self_fn, args_list)` ABI).
    /// 3. Compile the body — kit emits the args-list-walking prologue
    ///    and the capture-load loop; we lower the user code on top.
    /// 4. Allocate via `ClosureKit::make` — inline GC alloc IR, no
    ///    extern thunk required.
    fn lower_fn_expr(
        &mut self,
        fb: &mut FunctionBuilder,
        env: &mut Env,
        rest: u64,
    ) -> Value {
        let clauses = parse_clauses(rest, self.sym);

        // Union of free vars across all clauses (filtered by each
        // clause's own params), restricted to names the outer env has
        // bound. First-seen order — same order the kit reads them out
        // of the closure's varlen tail.
        let mut captures: Vec<u32> = Vec::new();
        for c in &clauses {
            let mut param_ids = c.params.fixed.clone();
            if let Some(r) = c.params.rest {
                param_ids.push(r);
            }
            let raw = {
                let sym_ref = &*self.sym;
                crate::freevars::free_vars(c.body, &param_ids, &|id| {
                    sym_ref.name(id).to_string()
                })
            };
            for id in raw {
                if env.lookup(id).is_some() && !captures.contains(&id) {
                    captures.push(id);
                }
            }
        }

        *self.anon_counter += 1;
        let name = format!("__lambda_{}", *self.anon_counter);
        // Body signature is always `(self_fn, args_list)` for the
        // ArgsList convention regardless of clause arity — the kit's
        // dispatcher picks the right clause at runtime from the list
        // count.
        let inner_fref = self.closures.declare_body(
            &mut self.dyn_module.module_builder,
            &name,
            BodyShape { fixed: 0, variadic: false, n_captures: captures.len() },
        );
        self.compile_closure_body_clauses(inner_fref, &clauses, &captures);

        // Arity word: min(arities), variadic if any clause is or if
        // multi-arity (so indirect callers accept any n ≥ min and
        // delegate exact matching to the inner dispatcher).
        let min_arity = clauses.iter().map(|c| c.params.min_arity()).min().unwrap_or(0);
        let any_variadic =
            clauses.iter().any(|c| c.params.is_variadic()) || clauses.len() > 1;
        let arity_word = encode_arity(min_arity, any_variadic) as i64;

        // Resolve captures from the outer env. Order matches what the
        // body prologue reads back.
        let cap_vals: Vec<Value> = captures
            .iter()
            .map(|&id| env.lookup(id).expect("capture not in outer env"))
            .collect();

        let live_pre = env.live_values();
        self.closures.make(
            fb,
            MakeClosure {
                body_ref: inner_fref,
                arity_word,
                captures: &cap_vals,
                extras: &[],
            },
            &live_pre,
        )
    }

    /// Compile a closure body (1 or N arity clauses) through ClosureKit.
    ///
    /// Body shape is delegated to the kit: ABI is `(self_fn, args_list)`,
    /// the kit emits the arity check (single-arity) or dispatch chain
    /// (multi-arity), walks the args list to bind params, and loads
    /// captures from the receiver. This frontend's only job is to copy
    /// the kit's `BoundBodyEnv` into the per-clause `Env` so symbol
    /// lookup resolves params and captures correctly.
    fn compile_closure_body_clauses(
        &mut self,
        fref: FuncRef,
        clauses: &[Clause],
        captures: &[u32],
    ) {
        let mut fb = self.dyn_module.module_builder.define_func(fref);

        if clauses.len() == 1 {
            let c = &clauses[0];
            let shape = BodyShape {
                fixed: c.params.fixed.len(),
                variadic: c.params.is_variadic(),
                n_captures: captures.len(),
            };
            let bound = self.closures.begin_body(&mut fb, shape);
            self.lower_kit_clause(&mut fb, &c.params, captures, c.body, bound);
            self.dyn_module.module_builder.finish_func(fref, fb);
            return;
        }

        let shapes: Vec<BodyShape> = clauses
            .iter()
            .map(|c| BodyShape {
                fixed: c.params.fixed.len(),
                variadic: c.params.is_variadic(),
                n_captures: captures.len(),
            })
            .collect();
        let dispatch = self.closures.begin_multi_arity_body(&mut fb, &shapes);
        for (i, clause) in clauses.iter().enumerate() {
            let bound = dispatch.begin_clause(self.closures, &mut fb, i);
            self.lower_kit_clause(&mut fb, &clause.params, captures, clause.body, bound);
        }
        self.dyn_module.module_builder.finish_func(fref, fb);
    }

    /// Bind a kit-produced `BoundBodyEnv` into our `Env` and lower a
    /// single clause's body. Handles loop-target setup so `recur` can
    /// retarget the kit's prologue loop_header.
    fn lower_kit_clause(
        &mut self,
        fb: &mut FunctionBuilder,
        params: &Params,
        captures: &[u32],
        body_forms: u64,
        bound: BoundBodyEnv,
    ) {
        let mut env = Env::new();
        for (i, &pid) in params.fixed.iter().enumerate() {
            env.bind(pid, bound.args[i]);
        }
        if let Some(rid) = params.rest {
            env.bind(
                rid,
                bound.rest.expect("variadic clause: kit must bind rest arg"),
            );
        }
        for (i, &cap_id) in captures.iter().enumerate() {
            env.bind(cap_id, bound.captures[i]);
        }
        self.loop_targets.push(LoopTarget {
            block: bound.recur_block,
            arity: params.min_arity() + if params.is_variadic() { 1 } else { 0 },
            prepend: vec![bound.self_fn],
            pack_args: true,
        });
        let result = self.lower_do(fb, &mut env, body_forms);
        fb.ret(result);
        self.loop_targets.pop();
    }

    fn lower_loop(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // (loop [n0 v0 n1 v1 …] body…) — like let, but the binding
        // names are the targets of any enclosed `recur`. At the IR
        // level we open a fresh basic block whose block params are the
        // bindings, jump to it with the init values, then lower the
        // body in a scope that binds each name to its block param.
        let bindings_form = v::first(args);
        let body_forms = v::rest(args);
        let pairs: Vec<u64> = crate::collections::seq_iter(bindings_form).collect();
        if pairs.len() % 2 != 0 {
            panic!("loop: bindings must come in name/value pairs");
        }
        let mut names: Vec<u32> = Vec::new();
        let mut init_vals: Vec<Value> = Vec::new();
        for chunk in pairs.chunks(2) {
            let name_v = chunk[0];
            if !v::is_sym_id(name_v) {
                panic!("loop: binding name must be a symbol");
            }
            names.push(v::as_sym_id(name_v));
            // Init values evaluate in the OUTER env (sequential let
            // semantics: later inits see earlier names — rebound on
            // each push to env so the second loop test below holds).
            let val = self.lower_expr(fb, env, chunk[1]);
            env.push();
            env.bind(v::as_sym_id(name_v), val);
            init_vals.push(val);
        }
        // Pop the temporary inner scopes we used for sequential let
        // semantics during init.
        for _ in 0..names.len() {
            env.pop();
        }

        let header = fb.create_block(&vec![Type::I64; names.len()]);
        fb.jump(header, &init_vals);
        fb.switch_to_block(header);

        // Bind each name to the corresponding block_param.
        env.push();
        for (i, &nid) in names.iter().enumerate() {
            env.bind(nid, fb.block_param(header, i));
        }

        // Push as the new innermost recur target (no prepend — `loop`
        // doesn't pass any hidden self argument; no list packing —
        // the loop header carries the bindings as direct block params).
        self.loop_targets.push(LoopTarget {
            block: header,
            arity: names.len(),
            prepend: Vec::new(),
            pack_args: false,
        });
        let result = self.lower_do(fb, env, body_forms);
        self.loop_targets.pop();
        env.pop();
        result
    }

    fn lower_recur(&mut self, fb: &mut FunctionBuilder, env: &mut Env, args: u64) -> Value {
        // Lower each user arg in the current env.
        let arg_forms: Vec<u64> = v::list_iter(args).collect();
        let arg_vals: Vec<Value> = arg_forms
            .iter()
            .map(|f| self.lower_expr(fb, env, *f))
            .collect();

        // Snapshot the target's relevant fields BEFORE we call any
        // helpers (build_args_list etc) that take `&mut self`.
        let target_block;
        let target_arity;
        let target_pack;
        let target_prepend: Vec<Value>;
        {
            let target = self
                .loop_targets
                .last()
                .expect("recur: no enclosing loop or fn body");
            target_block = target.block;
            target_arity = target.arity;
            target_pack = target.pack_args;
            target_prepend = target.prepend.clone();
        }
        if arg_vals.len() != target_arity {
            panic!(
                "recur: expected {} arg(s), got {}",
                target_arity,
                arg_vals.len()
            );
        }

        // Build the jump-args list. For list-ABI targets (closures
        // and variadic def-fns), pack the user args into a single
        // Clojure list, mirroring how the corresponding call site
        // packs. Then prepend any captured-into-target values
        // (e.g. self_fn for closures).
        let mut full_args: Vec<Value> = target_prepend;
        if target_pack {
            let list = self.build_args_list(fb, env, &arg_vals);
            full_args.push(list);
        } else {
            full_args.extend(arg_vals.iter().copied());
        }

        fb.jump(target_block, &full_args);

        // Switch to a fresh unreachable block so subsequent IR
        // emitted by lower_expr's caller has somewhere to land.
        let dead = fb.create_block(&[]);
        fb.switch_to_block(dead);
        // recur is non-returning from the perspective of the
        // enclosing scope (control jumped to the loop target);
        // surrounding `lower_try` should skip its prompt cleanup.
        self.last_expr_non_returning = true;
        fb.iconst(Type::I64, 0)
    }
}

// ── Compound-literal detection ──────────────────────────────────────

/// True if this vector contains at least one element that needs
/// runtime evaluation (a symbol, a list, or a nested vector that
/// itself needs evaluation). Pure-literal vectors (numbers,
/// keywords, strings, nil, all-literal nested vectors) can be
/// pinned to the literal pool as-is.
fn vector_needs_runtime_eval(vec: u64) -> bool {
    for elem in crate::collections::vector_iter(vec) {
        if needs_runtime_eval(elem) {
            return true;
        }
    }
    false
}

fn needs_runtime_eval(form: u64) -> bool {
    if v::is_sym_id(form) {
        return true;
    }
    if !v::is_ptr(form) {
        return false;
    }
    if crate::collections::is_list(form) {
        return true;
    }
    if crate::collections::is_vector(form) {
        return vector_needs_runtime_eval(form);
    }
    false
}

// ── Truthiness ──────────────────────────────────────────────────────

/// Truthy = (val != NIL) && (val != FALSE).
fn compute_truthy(fb: &mut FunctionBuilder, val: Value) -> Value {
    let nil_c = fb.iconst(Type::I64, v::NIL as i64);
    let false_c = fb.iconst(Type::I64, v::FALSE as i64);
    let ne_nil = fb.icmp(CmpOp::Ne, val, nil_c);
    let ne_false = fb.icmp(CmpOp::Ne, val, false_c);
    fb.and(ne_nil, ne_false)
}

// ── Environment ─────────────────────────────────────────────────────

pub struct Env {
    scopes: Vec<HashMap<u32, Value>>,
}

impl Env {
    pub fn new() -> Self {
        Env {
            scopes: vec![HashMap::new()],
        }
    }
    pub fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }
    pub fn pop(&mut self) {
        self.scopes.pop();
    }
    pub fn bind(&mut self, id: u32, v: Value) {
        self.scopes.last_mut().unwrap().insert(id, v);
    }
    pub fn lookup(&self, id: u32) -> Option<Value> {
        for s in self.scopes.iter().rev() {
            if let Some(&v) = s.get(&id) {
                return Some(v);
            }
        }
        None
    }

    /// Snapshot of every IR `Value` currently bound in any scope.
    /// Over-approximates the live set at any point — a few false
    /// positives (e.g. shadowed bindings) are harmless. Used to feed
    /// the live-arg list of `fb.safepoint(...)` so the GC's stack-map
    /// updates the spill slots of these values when it relocates.
    pub fn live_values(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            for &v in scope.values() {
                out.push(v);
            }
        }
        out
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}
