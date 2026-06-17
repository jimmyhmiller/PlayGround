//! Inferred effects.
//!
//! Every definition gets an **effect signature**: the set of primitive
//! effects it may perform, plus a record of which of its function-typed
//! parameters' effects it incurs (effect *polymorphism*, the analogue of
//! generic type variables). Effects are inferred bottom-up over the call
//! graph — the leaves are builtins, each classified once in
//! [`builtin_effect_sig`] — with a fixpoint over each strongly-connected
//! component for recursion (reusing [`crate::resolve::tarjan_scc`]).
//!
//! Because every reference is by content hash, a def's effect signature is
//! a pure function of its hash: computed once, cacheable forever next to
//! its typescheme.
//!
//! ## Effect polymorphism (stage 1)
//!
//! A higher-order function's effect depends on the functions passed to it:
//! `twice(f) = f() + f()` performs *whatever calling `f` performs*. We
//! capture this with `param_deps`: a bitmask of the def's parameter indices
//! whose latent effect is incurred. At a call site, each dependency is
//! resolved against the actual argument (`twice(loud)` → `{IO}`,
//! `twice(quiet)` → `∅`) — the same instantiate-at-the-call-site move the
//! generic type machinery already makes. Because effects form a join
//! lattice and this is a may-analysis, "solving" a dependency is just
//! union, never constraint solving.
//!
//! Not handled in stage 1 (sound, but over-approximated to `ALL`): a def
//! that *returns* an effectful closure (the effect would need to live on
//! the returned arrow — see the `FnType` effect-column work, stage 2).

use std::collections::HashMap;

use crate::ast::{Def, Expr, Pattern, Type};
use crate::hash::Hash;
use crate::resolve::{ResolvedModule, tarjan_scc};

// ─── EffectSet ──────────────────────────────────────────────────────

/// A set of primitive effects, as a bitset.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectSet(u16);

impl EffectSet {
    pub const EMPTY: EffectSet = EffectSet(0);
    /// External input/output (print / read externs).
    pub const IO: EffectSet = EffectSet(1 << 0);
    /// Network: a remote `at(node, ...)` call.
    pub const NET: EffectSet = EffectSet(1 << 1);
    /// Reads or writes node-resident `state`.
    pub const STATE: EffectSet = EffectSet(1 << 2);
    /// Reads or writes a local `Atom` cell (shared mutable state).
    pub const ATOM: EffectSet = EffectSet(1 << 3);
    /// Mutates an owned/passed `Array` or `Bytes` (local mutation).
    pub const MUT: EffectSet = EffectSet(1 << 4);
    /// Raw C call / `Ptr` op — opaque, assume the worst.
    pub const FFI: EffectSet = EffectSet(1 << 5);
    /// Every effect — the sound over-approximation for anything we can't
    /// see through (an unknown computed function value, an external def).
    pub const ALL: EffectSet = EffectSet(0b11_1111);

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn contains(self, other: EffectSet) -> bool {
        self.0 & other.0 == other.0
    }
    pub fn union(self, other: EffectSet) -> EffectSet {
        EffectSet(self.0 | other.0)
    }
    /// `self` with `mask`'s effects removed.
    pub fn without(self, mask: EffectSet) -> EffectSet {
        EffectSet(self.0 & !mask.0)
    }

    /// The individual effects in this set, as lowercase tokens (the inverse
    /// of [`from_tokens`](Self::from_tokens)). Empty ⇒ `[]` (i.e. `pure`).
    pub fn tokens(self) -> Vec<&'static str> {
        [
            (EffectSet::IO, "io"),
            (EffectSet::NET, "net"),
            (EffectSet::STATE, "state"),
            (EffectSet::ATOM, "atom"),
            (EffectSet::MUT, "mut"),
            (EffectSet::FFI, "ffi"),
        ]
        .into_iter()
        .filter(|(bit, _)| self.contains(*bit))
        .map(|(_, name)| name)
        .collect()
    }

    /// Parse a node effect policy: a comma/space/`|`-separated list of
    /// `io,net,state,atom,mut,ffi`, or the shorthands `all` (every effect)
    /// and `pure`/`none`/empty (no effects). Used to read a node's allowed
    /// effect set from config (`AI_LANG_AT_EFFECTS`). Unknown tokens error
    /// rather than being silently ignored (a typo'd policy must not widen
    /// access).
    pub fn from_tokens(s: &str) -> Result<EffectSet, String> {
        let mut set = EffectSet::EMPTY;
        for tok in s.split([',', ' ', '|', '\t']).filter(|t| !t.is_empty()) {
            set |= match tok.to_ascii_lowercase().as_str() {
                "all" => EffectSet::ALL,
                "pure" | "none" => EffectSet::EMPTY,
                "io" => EffectSet::IO,
                "net" => EffectSet::NET,
                "state" => EffectSet::STATE,
                "atom" => EffectSet::ATOM,
                "mut" => EffectSet::MUT,
                "ffi" => EffectSet::FFI,
                other => {
                    return Err(format!(
                        "unknown effect `{other}` (expected io|net|state|atom|mut|ffi|all|pure)"
                    ))
                }
            };
        }
        Ok(set)
    }
}

impl std::ops::BitOr for EffectSet {
    type Output = EffectSet;
    fn bitor(self, rhs: EffectSet) -> EffectSet {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for EffectSet {
    fn bitor_assign(&mut self, rhs: EffectSet) {
        self.0 |= rhs.0;
    }
}

impl std::fmt::Debug for EffectSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "pure");
        }
        let names = [
            (EffectSet::IO, "IO"),
            (EffectSet::NET, "Net"),
            (EffectSet::STATE, "State"),
            (EffectSet::ATOM, "Atom"),
            (EffectSet::MUT, "Mut"),
            (EffectSet::FFI, "FFI"),
        ];
        let mut first = true;
        write!(f, "{{")?;
        for (bit, name) in names {
            if self.contains(bit) {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{name}")?;
                first = false;
            }
        }
        write!(f, "}}")
    }
}

// ─── EffectSig ──────────────────────────────────────────────────────

/// A definition's inferred effect signature.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct EffectSig {
    /// Effects this def performs regardless of its arguments.
    pub concrete: EffectSet,
    /// Bitmask of parameter indices whose latent (when-called) effect this
    /// def incurs. Bit `j` set ⇒ calling this def runs whatever calling its
    /// `j`th (function-typed) parameter would run. This is the effect
    /// analogue of a generic type variable.
    pub param_deps: u64,
    /// (Stage 2) The latent effect of this def's RETURNED VALUE, when that
    /// value is a function — i.e. what *calling the result* incurs. For
    /// `make_runner(f) = || f()`, the body is pure to evaluate (`concrete`
    /// empty) but its result, when called, runs `f` — so `result_latent`
    /// records `dep(0)`. Lets a closure factory's result be analyzed
    /// precisely instead of over-approximated to `ALL`. Pure for defs that
    /// return data.
    pub result_latent: EffectSet,
    pub result_latent_deps: u64,
}

impl EffectSig {
    /// Unconditionally pure: no effects and no dependence on any argument.
    pub fn is_pure(&self) -> bool {
        self.concrete.is_empty() && self.param_deps == 0
    }
    /// Mobile = safe to ship to another thread OR node. Disqualified by raw
    /// FFI (no `Ptr` over a boundary) and by touching a local `Atom` (the
    /// cell doesn't exist elsewhere). `State` is fine (resolves to the
    /// destination's state); `Net`/`IO`/`Mut`/`Panic` run at the
    /// destination. Polymorphic over its arguments' mobility, so a def with
    /// `param_deps` is mobile only when given mobile arguments — callers
    /// check that at the call site.
    pub fn is_mobile(&self) -> bool {
        !self.concrete.contains(EffectSet::FFI) && !self.concrete.contains(EffectSet::ATOM)
    }
    /// Cacheable / idempotent for the `at` result cache: deterministic and
    /// free of external/observable effects. Local `Mut` (on copied
    /// values) is allowed.
    pub fn cacheable(&self) -> bool {
        self.param_deps == 0
            && self.concrete.without(EffectSet::MUT).is_empty()
    }
}

// ─── Outcome (the monoid the walk accumulates) ──────────────────────

/// The effect of evaluating (or of calling) an expression: a concrete
/// effect set plus a set of current-def parameter indices whose latent
/// effect is incurred. Unions form the lattice.
#[derive(Clone, Copy, Default)]
struct Outcome {
    eff: EffectSet,
    deps: u64,
}

impl Outcome {
    const PURE: Outcome = Outcome {
        eff: EffectSet::EMPTY,
        deps: 0,
    };
    fn eff(e: EffectSet) -> Outcome {
        Outcome { eff: e, deps: 0 }
    }
    /// Calling parameter `j` of the current def.
    fn dep(j: u32) -> Outcome {
        Outcome {
            eff: EffectSet::EMPTY,
            deps: 1u64 << j,
        }
    }
    const TOP: Outcome = Outcome {
        eff: EffectSet::ALL,
        deps: 0,
    };
    fn join(self, other: Outcome) -> Outcome {
        Outcome {
            eff: self.eff | other.eff,
            deps: self.deps | other.deps,
        }
    }
}

// ─── Builtin classification (the leaves) ────────────────────────────

/// The effect signature of a builtin, by its stable name. Returns the
/// concrete effects it performs plus a bitmask of argument positions whose
/// latent effect it incurs (e.g. `atom_swap`'s updater, `at`'s thunk,
/// `spawn`'s thunk).
pub fn builtin_effect_sig(name: &str) -> EffectSig {
    let pure = EffectSig::default();
    let just = |e: EffectSet| EffectSig {
        concrete: e,
        ..EffectSig::default()
    };
    let with_dep = |e: EffectSet, arg: u32| EffectSig {
        concrete: e,
        param_deps: 1u64 << arg,
        ..EffectSig::default()
    };

    // `at` ships with its Result/Failure hashes baked into the name.
    if name == "core/net.at" || crate::resolve::parse_at_builtin_name(name).is_some() {
        // at(node, thunk): network, and may cause the thunk's effects.
        return with_dep(EffectSet::NET, 1);
    }
    // `wire.decode` is also a hashed name.
    if name.starts_with("core/wire.decode") {
        return pure;
    }

    match name {
        // Pure scalar / value ops.
        n if n.starts_with("core/i64.")
            || n.starts_with("core/f64.")
            || n.starts_with("core/bool.")
            || n.starts_with("core/string.") => pure,
        // Hash / structural equality — pure.
        "core/hash.value" | "core/value.eq" => pure,
        // GC is value-transparent.
        "core/gc.collect" => pure,

        // Arrays / Bytes are owned values (copied across boundaries), so
        // they don't break mobility; reads/constructs are pure, in-place
        // writes are a local mutation.
        "core/array.new" | "core/array.new_prim" | "core/array.len" | "core/array.get" | "core/array.get_scalar" | "core/array.is_init" => pure,
        "core/array.set" | "core/array.set_scalar" => just(EffectSet::MUT),
        "core/bytes.new"
        | "core/bytes.len"
        | "core/bytes.get"
        | "core/bytes.from_string"
        | "core/bytes.concat"
        | "core/bytes.slice" => pure,
        "core/bytes.set" => just(EffectSet::MUT),

        // Constructing a cell is pure; reading/writing the shared cell is
        // the effect. `swap` also runs its updater closure (arg 1).
        "core/atom.new" => pure,
        "core/atom.load" => just(EffectSet::ATOM),
        "core/atom.swap" => with_dep(EffectSet::ATOM, 1),

        // Threads: spawning may cause the thunk's (arg 0) effects.
        "core/thread.spawn" | "core/thread.spawn_shared" => EffectSig {
            concrete: EffectSet::EMPTY,
            param_deps: 1,
            ..EffectSig::default()
        },
        "core/thread.join" => pure,

        // Wire codec: encode/decode are pure; invoke runs arbitrary shipped
        // code, so it's opaque.
        "core/wire.encode" => pure,
        "core/wire.invoke" | "core/wire.decode_fn1" => just(EffectSet::ALL),

        // Raw pointer / FFI memory.
        n if n.starts_with("core/ptr.") => just(EffectSet::FFI),

                // User `extern fn`s. The conventional stdlib I/O + pure-conversion
        // externs are classified precisely; everything else into C is
        // opaque → FFI. (A user could shadow these names; that's an
        // accepted stage-1 imprecision.)
        n if n.starts_with("ext/") => {
            let base = &n["ext/".len()..];
            match base {
                // Pure conversions — no effect.
                "int_to_string" | "string_to_int" | "string_is_int" => pure,
                // I/O and process-environment access.
                "print_int" | "print_string" | "println" | "print" | "eprintln"
                | "read_line" | "arg_count" | "get_arg" | "sleep_ms" => just(EffectSet::IO),
                // Any other C call is opaque.
                _ => just(EffectSet::FFI),
            }
        }

        // Anything unrecognized: be sound.
        _ => just(EffectSet::ALL),
    }
}

// ─── The inference pass ─────────────────────────────────────────────

struct Inferer<'a> {
    idx_of: &'a HashMap<Hash, usize>,
    /// Current effect estimate per in-module def (by index). Grows
    /// monotonically during the SCC fixpoint.
    sigs: &'a [EffectSig],
}

impl Inferer<'_> {
    /// The latent effect of `expr` *as a callable value* — what calling it
    /// would incur. `env` is the de Bruijn binder stack: `Some(outcome)` is
    /// a callable binder's latent effect, `None` a non-callable one.
    fn call_effect_of(&self, expr: &Expr, env: &[Option<Outcome>]) -> Outcome {
        match expr {
            Expr::LocalVar(i) => {
                let n = env.len();
                let idx = *i as usize;
                if idx >= n {
                    return Outcome::TOP;
                }
                env[n - 1 - idx].unwrap_or(Outcome::TOP)
            }
            Expr::TopRef(h) => match self.idx_of.get(h) {
                // A def passed as a value: if it's effect-polymorphic we
                // can't resolve its deps here, so be sound (TOP). Otherwise
                // its concrete effect is exact.
                Some(&j) => {
                    let s = self.sigs[j];
                    if s.param_deps == 0 {
                        Outcome::eff(s.concrete)
                    } else {
                        Outcome::TOP
                    }
                }
                None => Outcome::TOP,
            },
            Expr::BuiltinRef(name) => {
                let s = builtin_effect_sig(name);
                if s.param_deps == 0 {
                    Outcome::eff(s.concrete)
                } else {
                    Outcome::TOP
                }
            }
            Expr::Lambda { params, body } => {
                // The lambda's latent effect is its body's effect. Its own
                // params are pushed; a fn-typed lambda param is supplied by
                // whoever ultimately calls the lambda, so calling it is
                // unknown → TOP (sound). Captures resolve through `env`, so
                // a lambda that calls the current def's param propagates
                // that dependency.
                let mut inner = env.to_vec();
                for _ in params {
                    inner.push(Some(Outcome::TOP));
                }
                self.walk(body, &inner)
            }
            // (Stage 2) The value is the RESULT of calling `inner`; calling
            // that result incurs `inner`'s `result_latent`, resolved against
            // this call's arguments. E.g. `make_runner(loud)` evaluates to a
            // thunk whose call incurs `loud`'s effect.
            Expr::Call(inner, iargs) => match inner.as_ref() {
                Expr::TopRef(h) => match self.idx_of.get(h) {
                    Some(&j) => {
                        let s = self.sigs[j];
                        self.resolve(s.result_latent, s.result_latent_deps, iargs, env)
                    }
                    None => Outcome::TOP,
                },
                // A builtin returns data, never a function — calling its
                // result is nonsense, so pure.
                Expr::BuiltinRef(_) => Outcome::PURE,
                _ => Outcome::TOP,
            },
            // Thread the returned value out of binding/branching forms so a
            // closure produced through a `let`/`if`/`match` is still seen.
            Expr::Let { value, body } => {
                let entry = self.binder_for(value, env);
                let mut inner = env.to_vec();
                inner.push(entry);
                self.call_effect_of(body, &inner)
            }
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => self
                .call_effect_of(then_branch, env)
                .join(self.call_effect_of(else_branch, env)),
            Expr::Match { arms, .. } => {
                let mut out = Outcome::PURE;
                for arm in arms {
                    let binds = count_pattern_bindings(&arm.pattern);
                    let mut inner = env.to_vec();
                    for _ in 0..binds {
                        inner.push(None);
                    }
                    out = out.join(self.call_effect_of(&arm.body, &inner));
                }
                out
            }
            Expr::Defer { body, .. } => self.call_effect_of(body, env),
            // Anything else (a field read, a struct, …): can't see through.
            _ => Outcome::TOP,
        }
    }

    /// Resolve a latent effect `(eff, deps)` whose `deps` reference some
    /// callee's parameters, against the actual `args` at a call site:
    /// union `eff` with the latent effect of each argument in a dep slot.
    fn resolve(
        &self,
        eff: EffectSet,
        deps: u64,
        args: &[Expr],
        env: &[Option<Outcome>],
    ) -> Outcome {
        let mut out = Outcome::eff(eff);
        let mut d = deps;
        while d != 0 {
            let j = d.trailing_zeros() as usize;
            d &= d - 1;
            if let Some(arg) = args.get(j) {
                out = out.join(self.call_effect_of(arg, env));
            }
        }
        out
    }

    /// The effect of *evaluating* `expr`.
    fn walk(&self, expr: &Expr, env: &[Option<Outcome>]) -> Outcome {
        match expr {
            Expr::IntLit(_)
            | Expr::FloatLit(_)
            | Expr::BoolLit(_)
            | Expr::StringLit(_)
            | Expr::LocalVar(_)
            | Expr::TopRef(_)
            | Expr::BuiltinRef(_)
            | Expr::SelfRef(_) => Outcome::PURE,

            // Reading node-resident state.
            Expr::StateRef(_) | Expr::StateSelfRef(_) => Outcome::eff(EffectSet::STATE),

            Expr::Call(callee, args) => {
                // Evaluating the callee and each argument.
                let mut out = self.walk(callee, env);
                for a in args {
                    out = out.join(self.walk(a, env));
                }
                // The call itself.
                out.join(self.call_contribution(callee, args, env))
            }

            // Creating a closure is pure; its body's effects happen when it
            // is *called* (handled at call sites / `call_effect_of`).
            Expr::Lambda { .. } => Outcome::PURE,

            Expr::Let { value, body } => {
                let v = self.walk(value, env);
                // Bind the let var; if it's a callable value, record its
                // latent effect so a later call resolves precisely.
                let entry = self.binder_for(value, env);
                let mut inner = env.to_vec();
                inner.push(entry);
                v.join(self.walk(body, &inner))
            }

            Expr::Defer { cleanup, body } => self.walk(cleanup, env).join(self.walk(body, env)),

            Expr::StructNew { fields, .. } => {
                let mut out = Outcome::PURE;
                for fe in fields {
                    out = out.join(self.walk(fe, env));
                }
                out
            }
            Expr::Field { base, .. } => self.walk(base, env),
            Expr::EnumNew { payload, .. } => match payload {
                Some(p) => self.walk(p, env),
                None => Outcome::PURE,
            },

            Expr::Match { scrutinee, arms } => {
                let mut out = self.walk(scrutinee, env);
                for arm in arms {
                    let binds = count_pattern_bindings(&arm.pattern);
                    let mut inner = env.to_vec();
                    for _ in 0..binds {
                        // Pattern bindings are data (or, if a fn-typed
                        // payload, supplied externally → TOP when called).
                        inner.push(None);
                    }
                    out = out.join(self.walk(&arm.body, &inner));
                }
                out
            }

            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => self
                .walk(cond, env)
                .join(self.walk(then_branch, env))
                .join(self.walk(else_branch, env)),

            Expr::Try { expr, .. } => self.walk(expr, env),
        }
    }

    /// The latent effect to record for a `let`-bound value, so that calling
    /// the bound name later resolves precisely.
    fn binder_for(&self, value: &Expr, env: &[Option<Outcome>]) -> Option<Outcome> {
        match value {
            // `Call` included so a let-bound closure-factory result
            // (`let r = make_runner(loud)`) carries its latent effect.
            Expr::TopRef(_)
            | Expr::BuiltinRef(_)
            | Expr::Lambda { .. }
            | Expr::LocalVar(_)
            | Expr::Call(..) => Some(self.call_effect_of(value, env)),
            _ => None,
        }
    }

    /// The effect contributed by *the call itself* (beyond evaluating the
    /// subexpressions): the callee's concrete effect plus, for each of its
    /// dependent parameter positions, the latent effect of the argument
    /// passed there.
    fn call_contribution(
        &self,
        callee: &Expr,
        args: &[Expr],
        env: &[Option<Outcome>],
    ) -> Outcome {
        let resolve_deps = |concrete: EffectSet, deps: u64| -> Outcome {
            let mut out = Outcome::eff(concrete);
            let mut d = deps;
            while d != 0 {
                let j = d.trailing_zeros() as usize;
                d &= d - 1;
                if let Some(arg) = args.get(j) {
                    out = out.join(self.call_effect_of(arg, env));
                }
            }
            out
        };

        match callee {
            Expr::BuiltinRef(name) => {
                let s = builtin_effect_sig(name);
                resolve_deps(s.concrete, s.param_deps)
            }
            Expr::TopRef(h) => match self.idx_of.get(h) {
                Some(&j) => {
                    let s = self.sigs[j];
                    resolve_deps(s.concrete, s.param_deps)
                }
                // External def we can't see: sound over-approximation.
                None => Outcome::TOP,
            },
            // Calling a bound name (a parameter or a let-bound function),
            // an immediately-applied lambda, or the result of another call
            // (a closure factory) / branch — `call_effect_of` sees through
            // all of these.
            _ => self.call_effect_of(callee, env),
        }
    }
}

/// Number of locals a pattern binds (each becomes an innermost binder in
/// the arm body's environment).
fn count_pattern_bindings(p: &Pattern) -> usize {
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => match payload {
            Some(inner) => count_pattern_bindings(inner),
            None => 0,
        },
    }
}

/// Is a parameter type a function type (so calling it carries an effect)?
fn is_fn_type(t: &Type) -> bool {
    matches!(t, Type::FnType { .. })
}

/// The body + params of a def that has effects to infer (`Fn`, `State`).
fn def_body(def: &Def) -> Option<(&[Type], &Expr)> {
    match def {
        Def::Fn { params, body, .. } => Some((params, body)),
        Def::State { init, .. } => Some((&[], init)),
        Def::Struct { .. } | Def::Enum { .. } => None,
    }
}

/// Infer an [`EffectSig`] for every `Fn`/`State` def in the module.
///
/// Closed-world over the module: a `TopRef` to a def not present here is
/// treated as `ALL` (sound). Within the module, recursion is resolved by a
/// monotone fixpoint over each strongly-connected component of the call
/// graph, processed callees-first (the order `tarjan_scc` yields).
pub fn infer_effects(module: &ResolvedModule) -> HashMap<Hash, EffectSig> {
    // Index the defs that have bodies.
    let mut idx_of: HashMap<Hash, usize> = HashMap::new();
    let mut bodies: Vec<(&[Type], &Expr)> = Vec::new();
    let mut hashes: Vec<Hash> = Vec::new();
    for d in &module.defs {
        if let Some(b) = def_body(&d.def) {
            idx_of.insert(d.hash, bodies.len());
            bodies.push(b);
            hashes.push(d.hash);
        }
    }
    let n = bodies.len();

    // Call-graph adjacency: edge i → j if def i's body references def j.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, (_, body)) in bodies.iter().enumerate() {
        let mut refs: Vec<u32> = Vec::new();
        collect_top_refs(body, &idx_of, &mut refs);
        refs.sort_unstable();
        refs.dedup();
        adj[i] = refs;
    }
    // SCCs in reverse-topological order (callees before callers).
    let sccs = tarjan_scc(&adj);

    let mut sigs: Vec<EffectSig> = vec![EffectSig::default(); n];

    // Pre-seed each def's env-shape: which params are function-typed.
    let fn_params: Vec<Vec<bool>> = bodies
        .iter()
        .map(|(params, _)| params.iter().map(is_fn_type).collect())
        .collect();

    for scc in &sccs {
        // Fixpoint within the component (monotone → terminates).
        loop {
            let mut changed = false;
            for &node in scc {
                let i = node as usize;
                let inf = Inferer {
                    idx_of: &idx_of,
                    sigs: &sigs,
                };
                // Initial env: the def's parameters. A fn-typed param `j`
                // contributes `dep(j)` when called; others are non-callable.
                let mut env: Vec<Option<Outcome>> = Vec::with_capacity(fn_params[i].len());
                for (j, &is_fn) in fn_params[i].iter().enumerate() {
                    env.push(if is_fn {
                        Some(Outcome::dep(j as u32))
                    } else {
                        None
                    });
                }
                let out = inf.walk(bodies[i].1, &env);
                // (Stage 2) The latent effect of the value this def returns,
                // when it is a function: what calling the result incurs.
                let rl = inf.call_effect_of(bodies[i].1, &env);
                let next = EffectSig {
                    concrete: out.eff,
                    param_deps: out.deps,
                    result_latent: rl.eff,
                    result_latent_deps: rl.deps,
                };
                if next != sigs[i] {
                    sigs[i] = next;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    hashes
        .into_iter()
        .enumerate()
        .map(|(i, h)| (h, sigs[i]))
        .collect()
}

/// Collect indices of in-module defs referenced (via `TopRef`) anywhere in
/// `e`.
fn collect_top_refs(e: &Expr, idx_of: &HashMap<Hash, usize>, out: &mut Vec<u32>) {
    match e {
        Expr::TopRef(h) => {
            if let Some(&j) = idx_of.get(h) {
                out.push(j as u32);
            }
        }
        Expr::Call(c, args) => {
            collect_top_refs(c, idx_of, out);
            for a in args {
                collect_top_refs(a, idx_of, out);
            }
        }
        Expr::Lambda { body, .. } => collect_top_refs(body, idx_of, out),
        Expr::Let { value, body } => {
            collect_top_refs(value, idx_of, out);
            collect_top_refs(body, idx_of, out);
        }
        Expr::Defer { cleanup, body } => {
            collect_top_refs(cleanup, idx_of, out);
            collect_top_refs(body, idx_of, out);
        }
        Expr::StructNew { fields, .. } => {
            for fe in fields {
                collect_top_refs(fe, idx_of, out);
            }
        }
        Expr::Field { base, .. } => collect_top_refs(base, idx_of, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_top_refs(p, idx_of, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_top_refs(scrutinee, idx_of, out);
            for arm in arms {
                collect_top_refs(&arm.body, idx_of, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_top_refs(cond, idx_of, out);
            collect_top_refs(then_branch, idx_of, out);
            collect_top_refs(else_branch, idx_of, out);
        }
        Expr::Try { expr, .. } => collect_top_refs(expr, idx_of, out),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    /// Resolve a standalone module and infer effects, returned by def name.
    fn infer(src: &str) -> HashMap<String, EffectSig> {
        let m = parse_module(src).expect("parse");
        let r = resolve_module(&m).expect("resolve");
        let by_hash = infer_effects(&r);
        r.defs
            .iter()
            .filter_map(|d| by_hash.get(&d.hash).map(|s| (d.name.clone(), *s)))
            .collect()
    }

    #[test]
    fn pure_arithmetic_is_pure() {
        let e = infer("def add1(x: Int) -> Int = x + 1");
        assert!(e["add1"].is_pure(), "got {:?}", e["add1"]);
        assert!(e["add1"].is_mobile());
        assert!(e["add1"].cacheable());
    }

    #[test]
    fn io_extern_is_io() {
        let e = infer(
            "
            extern fn print_int(n: Int) -> Int
            def shout(x: Int) -> Int = print_int(x)
        ",
        );
        assert_eq!(e["shout"].concrete, EffectSet::IO);
        assert!(!e["shout"].is_pure());
        assert!(!e["shout"].cacheable(), "IO is not cacheable");
        // IO is fine to run at the destination, so still mobile.
        assert!(e["shout"].is_mobile());
    }

    #[test]
    fn atom_touch_is_not_mobile() {
        let e = infer(
            "
            def reader(a: Atom<Int>) -> Int = deref(a)
            def deref<T>(a: Atom<T>) -> T = atom_load(a)
        ",
        );
        assert!(e["reader"].concrete.contains(EffectSet::ATOM));
        assert!(!e["reader"].is_mobile(), "a local Atom can't be shipped");
        assert!(!e["reader"].cacheable());
    }


    #[test]
    fn recursion_reaches_fixpoint() {
        let e = infer(
            "
            extern fn print_int(n: Int) -> Int
            def countdown(n: Int) -> Int =
                if n == 0 { 0 } else { let _p = print_int(n); countdown(n - 1) }
        ",
        );
        assert_eq!(e["countdown"].concrete, EffectSet::IO);
    }

    /// Effect polymorphism: a HOF carries a dependency on its function
    /// parameter, resolved per call site.
    #[test]
    fn hof_effect_is_polymorphic() {
        let e = infer(
            "
            extern fn print_int(n: Int) -> Int
            def twice(f: fn(Int) -> Int, x: Int) -> Int = f(x) + f(x)
            def loud(x: Int) -> Int = print_int(x)
            def quiet(x: Int) -> Int = x + 1
            def use_loud(x: Int) -> Int = twice(loud, x)
            def use_quiet(x: Int) -> Int = twice(quiet, x)
        ",
        );
        // `twice` itself is pure-bodied but depends on param 0.
        assert!(e["twice"].concrete.is_empty());
        assert_eq!(e["twice"].param_deps, 0b1, "depends on param 0 (f)");
        assert!(!e["twice"].is_pure(), "polymorphic, not unconditionally pure");

        // The dependency is resolved at each call site.
        assert_eq!(e["use_loud"].concrete, EffectSet::IO, "twice(loud) is IO");
        assert!(e["use_loud"].param_deps == 0);
        assert!(e["use_quiet"].is_pure(), "twice(quiet) is pure");
    }

    /// Runs over the real stdlib (~hundreds of defs) plus a small program:
    /// the SCC fixpoint must terminate and the results must be sane.
    #[test]
    fn infers_over_full_stdlib() {
        let src = format!(
            "{}\n{}",
            crate::stdlib::SOURCE,
            "
            def par() -> Int = join(spawn(|| 1 + 2))
            def noisy() -> Int = println_int(7)
            "
        );
        let e = infer(&src);

        // `spawn` is effect-polymorphic: pure-bodied, depends on its thunk.
        assert!(e["spawn"].concrete.is_empty(), "spawn body is pure-bodied");
        assert!(
            e["spawn"].param_deps != 0,
            "spawn depends on its thunk argument"
        );

        // Running a PURE computation in parallel is itself pure: the thunk
        // `|| 1 + 2` has no effects, so `join(spawn(...))` resolves to pure.
        assert!(e["par"].is_pure(), "par = {:?}", e["par"]);
        assert!(e["par"].is_mobile());

        // A def that prints carries IO.
        assert!(
            e["noisy"].concrete.contains(EffectSet::IO),
            "noisy = {:?}",
            e["noisy"]
        );
        assert!(!e["noisy"].cacheable());
    }

    /// Stage 2: a closure FACTORY's result carries the right latent effect,
    /// so calling the returned closure is analyzed precisely (not TOP).
    #[test]
    fn returned_closure_effect_is_precise() {
        let e = infer(
            "
            extern fn print_int(n: Int) -> Int
            def loud(x: Int) -> Int = print_int(x)
            def quiet(x: Int) -> Int = x + 1
            def make_runner(f: fn(Int) -> Int, x: Int) -> fn() -> Int = || f(x)
            def run_loud(x: Int) -> Int = make_runner(loud, x)()
            def run_quiet(x: Int) -> Int = make_runner(quiet, x)()
            def via_let(x: Int) -> Int = { let r = make_runner(loud, x); r() }
        ",
        );
        // make_runner is pure to CALL (it just builds a closure)...
        assert!(
            e["make_runner"].concrete.is_empty(),
            "building a closure is pure: {:?}",
            e["make_runner"]
        );
        // ...but its RESULT, when called, incurs param 0 (f)'s effect.
        assert_eq!(
            e["make_runner"].result_latent_deps, 0b1,
            "result depends on param 0 (f): {:?}",
            e["make_runner"]
        );
        // Calling the produced closure resolves precisely per call site.
        assert_eq!(e["run_loud"].concrete, EffectSet::IO, "run_loud is IO");
        assert!(e["run_quiet"].is_pure(), "run_quiet is pure");
        // Same precision when the closure flows through a `let`.
        assert_eq!(e["via_let"].concrete, EffectSet::IO, "via_let is IO");
    }

    #[test]
    fn nested_hof_propagates_dependency() {
        // `apply_twice` passes its own param into `twice`'s dep slot, so it
        // must itself become dependent on that param.
        let e = infer(
            "
            def twice(f: fn(Int) -> Int, x: Int) -> Int = f(x) + f(x)
            def apply_twice(g: fn(Int) -> Int, y: Int) -> Int = twice(g, y)
        ",
        );
        assert!(e["apply_twice"].concrete.is_empty());
        assert_eq!(
            e["apply_twice"].param_deps, 0b1,
            "dependency propagated to param 0 (g)"
        );
    }
}
