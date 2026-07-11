//! Clojure -> `Ir`, DIRECTLY. This frontend does NOT ride the toolkit's s-expr
//! `analyze`/special-forms — it compiles fully-expanded Clojure forms to the
//! toolkit's neutral `Ir` itself, owning name resolution, its own core forms
//! (`fn`/`let`/`if`/`do`/`def`/`quote`/`set!`/`defmethod`), prim mapping, and
//! protocol dispatch. The toolkit's job starts at `Ir`.

use std::collections::HashMap;
use std::sync::Arc;

use microlang::ir::{Ir, Prim};
use microlang::value::Sym;
use microlang::{Runtime, Val, ValueModel};

/// Compile-time state that persists across top-level forms.
pub struct Compiler {
    /// Lexical scope, innermost frame last (like `analyze`'s, but frontend-owned).
    scope: Vec<Vec<Sym>>,
    /// Protocol method names seen so far -> a call to one is a dispatch site.
    methods: std::collections::HashSet<Sym>,
    /// Monotonic dispatch-site id.
    site: usize,
    /// Monotonic `.-field` inline-cache site id.
    field_site: usize,
    /// The names THIS frontend treats as primitives (its choice, not the
    /// toolkit's). User-facing `+`/`first`/… are ordinary globals in
    /// `clojure.core`; only these low-level names map to a `Prim`.
    prims: HashMap<Sym, Prim>,
    /// Namespace state (frontend-owned; the toolkit's globals are a flat `Sym`
    /// pool, so a user-ns var `foo/x` is simply the interned sym `"foo/x"`).
    ns: NsState,
}

/// Namespace resolution state. `clojure.core` + the ported cljs types load in
/// `core_mode` (defs are BARE — the flat core space — and their names recorded in
/// `core_defs`); user code then runs with `core_mode = false`, qualifying its own
/// defs into `current_ns` and resolving bare refs own-def → refer → core.
#[derive(Default)]
struct NsState {
    current: String,
    core_mode: bool,
    /// Force the NEXT `def` to stay bare regardless of ns — used for macro fns,
    /// which live in the flat space keyed by short name so the expander finds them.
    bare_def: bool,
    core_defs: std::collections::HashSet<String>,
    ns_defs: HashMap<String, std::collections::HashSet<String>>,
    /// per-ns: alias -> real namespace name
    aliases: HashMap<String, HashMap<String, String>>,
    /// per-ns: short name -> fully-qualified sym string
    refers: HashMap<String, HashMap<String, String>>,
}

impl Compiler {
    pub fn new<M: ValueModel>(rt: &mut Runtime<M>) -> Self {
        use Prim::*;
        let mut prims = HashMap::new();
        for (name, p) in [
            ("%add", Add), ("%sub", Sub), ("%mul", Mul), ("%lt", Lt), ("%num-eq", Eq),
            ("%quot", Quot), ("%rem", Rem), ("%mod", Mod), ("%str-cat", StrCat), ("%str-of", StrOf),
            ("%apply", Apply),
            // Array substrate + bitwise ops for in-language persistent structures.
            ("%make-array", MakeArray), ("%aclone", AClone), ("%alength", VectorLen),
            ("%aget", VectorRef), ("%anew", Vector),
            ("%bit-and", BitAnd), ("%bit-or", BitOr), ("%bit-xor", BitXor),
            ("%bit-shl", BitShl), ("%bit-shr", BitShr), ("%bit-count", BitCount),
            ("%register-fields", RegisterFields), ("%field-by-name", FieldByName), ("%hash", Hash),
            ("%first", First), ("%rest", Rest), ("%cons", Cons),
            ("record", Record), ("field", Field), ("type-of", TypeOf), ("nfields", NFields), ("throw", Throw),
            ("nil?", IsNil), ("list", List), ("println", Println), ("gc", Gc),
            // Mutable 1-slot cell backing atoms (a real mutable array, unlike the
            // immutable list-backed clojure `vector`).
            ("%cell", Vector), ("%cell-ref", VectorRef), ("%cell-set!", VectorSet),
            // Real OS threads: spawn a thunk on a worker; await joins it.
            ("%spawn", Spawn), ("%await", Await),
            // Atoms: real cross-thread compare-and-set.
            ("%atom-new", AtomNew), ("%atom-get", AtomGet), ("%atom-set", AtomSet), ("%atom-cas", AtomCas),
        ] {
            prims.insert(rt.intern(name), p);
        }
        Compiler {
            scope: Vec::new(),
            methods: std::collections::HashSet::new(),
            site: 0,
            field_site: 0,
            prims,
            // core.clj + the ported cljs types load first, in the flat core space.
            ns: NsState { current: "clojure.core".to_string(), core_mode: true, ..NsState::default() },
        }
    }

    /// Called by `run` once clojure.core + the cljs types are loaded: subsequent
    /// (user) code qualifies its defs into `user` (or whatever `ns`/`in-ns` sets).
    pub fn end_core_load(&mut self) {
        self.ns.core_mode = false;
        self.ns.current = "user".to_string();
    }

    /// Resolve a symbol the way a reference would, per the current namespace —
    /// exposed so the frontend can locate a macro's fn global (it may have been
    /// defined into the current ns).
    pub fn resolve_ref<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym) -> Sym {
        self.resolve_global(rt, s)
    }

    /// Mark the next `def` as bare (flat space) — used by the macro-defining path,
    /// since macros are found by short name in the expander's flat macro table.
    pub fn next_def_bare(&mut self) {
        self.ns.bare_def = true;
    }

    /// `(ns foo …)` / `(in-ns 'foo)` — switch the current namespace.
    pub fn set_ns(&mut self, name: &str) {
        self.ns.current = name.to_string();
        self.ns.ns_defs.entry(name.to_string()).or_default();
    }

    /// `[foo :as bar]` / `(alias 'bar 'foo)` — `bar/x` now resolves to `foo/x`.
    pub fn add_alias(&mut self, alias: &str, real: &str) {
        let ns = self.ns.current.clone();
        self.ns.aliases.entry(ns).or_default().insert(alias.to_string(), real.to_string());
    }

    /// `[foo :refer [x]]` — bare `x` now resolves to the fully-qualified `foo/x`.
    pub fn add_refer(&mut self, short: &str, fq: &str) {
        let ns = self.ns.current.clone();
        self.ns.refers.entry(ns).or_default().insert(short.to_string(), fq.to_string());
    }

    /// Compile one fully-expanded top-level form to `Ir`.
    pub fn compile<M: ValueModel>(&mut self, rt: &mut Runtime<M>, form: u64) -> Ir {
        match rt.decode(form) {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => Ir::Const(rt.intern_const(form)),
            Val::Sym(s) => match self.resolve_local(s) {
                Some((up, idx)) => Ir::Local { up, idx },
                None => Ir::Global(self.resolve_global(rt, s)),
            },
            Val::Ref(_) => {
                if rt.as_cons(form).is_some() {
                    self.compile_list(rt, form)
                } else {
                    // keyword / string / char / quoted-collection literal: self-eval
                    Ir::Const(rt.intern_const(form))
                }
            }
        }
    }

    /// Resolve a REFERENCED global symbol to the interned sym its `Ir::Global`
    /// should read, per the current namespace. `a/b` resolves the ns/alias part;
    /// a bare name resolves own-def → refer → core → own-ns (forward ref).
    fn resolve_global<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym) -> Sym {
        // Prims (`%add`, …) and protocol-method names live in the flat space
        // (dispatch is by short name), never as ns-qualified globals.
        if self.prims.contains_key(&s) || self.methods.contains(&s) {
            return s;
        }
        let name = rt.sym_name(s);
        // Qualified `a/b` (but not the bare division op `/`).
        if name.len() > 1 {
            if let Some(slash) = name.find('/') {
                let (left, right) = (&name[..slash], &name[slash + 1..]);
                if !right.is_empty() {
                    let real = self
                        .ns
                        .aliases
                        .get(&self.ns.current)
                        .and_then(|a| a.get(left))
                        .map(String::as_str)
                        .unwrap_or(left);
                    // clojure.core is the flat space -> its members are bare.
                    if real == "clojure.core" {
                        return rt.intern(right);
                    }
                    return rt.intern(&format!("{real}/{right}"));
                }
            }
        }
        if self.ns.core_mode {
            return s;
        }
        let ns = &self.ns.current;
        if self.ns.ns_defs.get(ns).is_some_and(|d| d.contains(name)) {
            return rt.intern(&format!("{ns}/{name}")); // own def (shadows core)
        }
        if let Some(fq) = self.ns.refers.get(ns).and_then(|r| r.get(name)) {
            return rt.intern(fq);
        }
        if self.ns.core_defs.contains(name) {
            return s; // auto-referred clojure.core
        }
        // Unknown bare name: assume the current ns (forward ref to a var defined
        // later here, or an as-yet-unbound var).
        rt.intern(&format!("{ns}/{name}"))
    }

    /// Resolve a DEFINED name to the interned sym `Ir::Def` should write, and record
    /// it so later bare references in this ns resolve to it.
    fn def_name<M: ValueModel>(&mut self, rt: &Runtime<M>, raw: Sym) -> Sym {
        let bare = std::mem::take(&mut self.ns.bare_def); // consume unconditionally
        if rt.sym_name(raw).contains('/') {
            return self.resolve_global(rt, raw); // explicitly qualified def
        }
        if self.ns.core_mode {
            let n = rt.sym_name(raw).to_string();
            self.ns.core_defs.insert(n);
            return raw;
        }
        if bare {
            return raw;
        }
        let name = rt.sym_name(raw).to_string();
        let ns = self.ns.current.clone();
        self.ns.ns_defs.entry(ns.clone()).or_default().insert(name.clone());
        rt.intern(&format!("{ns}/{name}"))
    }

    fn resolve_local(&self, sym: Sym) -> Option<(u16, u16)> {
        for (up, frame) in self.scope.iter().rev().enumerate() {
            if let Some(idx) = frame.iter().rposition(|&s| s == sym) {
                return Some((up as u16, idx as u16));
            }
        }
        None
    }

    fn name<M: ValueModel>(&self, rt: &Runtime<M>, bits: u64) -> Option<Sym> {
        match rt.decode(bits) {
            Val::Sym(s) => Some(s),
            _ => None,
        }
    }

    fn compile_list<M: ValueModel>(&mut self, rt: &mut Runtime<M>, form: u64) -> Ir {
        let items = rt.list_to_vec(form);
        // A symbolic head may be a core special form, a prim, a method, or a call.
        if let Some(hs) = self.name(rt, items[0]) {
            match rt.sym_name(hs) {
                "quote" => return Ir::Const(rt.intern_const(items[1])),
                "if" => {
                    let c = self.compile(rt, items[1]);
                    let t = self.compile(rt, items[2]);
                    let e = if items.len() > 3 {
                        self.compile(rt, items[3])
                    } else {
                        let nil = rt.encode(Val::Nil);
                        Ir::Const(rt.intern_const(nil))
                    };
                    return Ir::If(Box::new(c), Box::new(t), Box::new(e));
                }
                "do" => return Ir::Do(items[1..].iter().map(|&f| self.compile(rt, f)).collect()),
                "def" => {
                    let raw = self.name(rt, items[1]).expect("def: name must be a symbol");
                    // Resolve+record the name FIRST so a self-reference inside the
                    // init (e.g. `(def x (fn [] x))`) resolves to this same var.
                    let n = self.def_name(rt, raw);
                    // value-less `(def x)` declares an unbound var (bound to nil).
                    let init = if items.len() > 2 {
                        self.compile(rt, items[2])
                    } else {
                        let nil = rt.encode(Val::Nil);
                        Ir::Const(rt.intern_const(nil))
                    };
                    return Ir::Def { name: n, init: Box::new(init) };
                }
                "fn" | "fn*" => return self.compile_fn(rt, &items),
                "let" | "let*" => return self.compile_let(rt, &items),
                "try*" => return self.compile_try(rt, &items),
                "set!" => {
                    let n = self.name(rt, items[1]).expect("set!: target must be a symbol");
                    let val = Box::new(self.compile(rt, items[2]));
                    return match self.resolve_local(n) {
                        Some((up, idx)) => Ir::SetLocal { up, idx, val },
                        None => Ir::SetGlobal { name: self.resolve_global(rt, n), val },
                    };
                }
                "-proto-method" => {
                    let m = self.name(rt, items[1]).expect("defmethod: method name");
                    let ty = self.name(rt, items[2]).expect("defmethod: type name");
                    self.methods.insert(m);
                    let imp = self.compile(rt, items[3]);
                    return Ir::DefMethod { name: m, ty, imp: Box::new(imp) };
                }
                _ => {
                    // A binding shadows a prim; a prim is otherwise its default.
                    let shadowed = self.resolve_local(hs).is_some() || rt.global_defined(hs);
                    if !shadowed {
                        if let Some(&p) = self.prims.get(&hs) {
                            // `(%field-by-name obj (quote field))` -> an inline-cached
                            // FieldGet; the field name is a compile-time constant.
                            if let Prim::FieldByName = p {
                                if let Some(field) = self.quoted_sym(rt, items[2]) {
                                    let obj = Box::new(self.compile(rt, items[1]));
                                    let site = self.fresh_field_site();
                                    return Ir::FieldGet { site, field, obj };
                                }
                            }
                            let args = items[1..].iter().map(|&f| self.compile(rt, f)).collect();
                            return Ir::Prim(p, args);
                        }
                    }
                    if self.methods.contains(&hs) {
                        let site = self.fresh_site();
                        let args = items[1..].iter().map(|&f| self.compile(rt, f)).collect();
                        return Ir::Dispatch { site, method: hs, args };
                    }
                }
            }
        }
        // ordinary call: (f args...)
        let f = self.compile(rt, items[0]);
        let args = items[1..].iter().map(|&a| self.compile(rt, a)).collect();
        Ir::Call(Box::new(f), args)
    }

    fn compile_fn<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let params = rt.list_to_vec(items[1]);
        let mut names = Vec::new();
        let mut variadic = false;
        let mut i = 0;
        while i < params.len() {
            let s = self.name(rt, params[i]).expect("fn: params must be symbols");
            if rt.sym_name(s) == "&" {
                let rest = self.name(rt, params[i + 1]).expect("fn: variadic param");
                names.push(rest);
                variadic = true;
                break;
            }
            names.push(s);
            i += 1;
        }
        let nparams = if variadic { names.len() - 1 } else { names.len() };
        self.scope.push(names);
        let body: Vec<Ir> = items[2..].iter().map(|&f| self.compile(rt, f)).collect();
        self.scope.pop();
        Ir::Lambda { nparams, variadic, body: Arc::new(Ir::Do(body)) }
    }

    /// `(try* body EXC dispatch finally)` — the fixed shape the expander lowers
    /// `try`/`catch`/`finally` to (see `desugar_try`). `body`/`finally` compile in
    /// the current scope; `dispatch` compiles with `EXC` bound in a fresh frame, so
    /// the thrown value lands at `Local{up:0,idx:0}` (matching `Ir::Try`). A `nil`
    /// `EXC`/`dispatch` means no catch; a `nil` `finally` means no finally.
    fn compile_try<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let body = Box::new(self.compile(rt, items[1]));
        let catch = match self.name(rt, items[2]) {
            Some(exc) => {
                self.scope.push(vec![exc]);
                let handler = self.compile(rt, items[3]);
                self.scope.pop();
                Some(Box::new(handler))
            }
            None => None, // items[2] is nil -> no catch clause
        };
        let finally = if matches!(rt.decode(items[4]), Val::Nil) {
            None
        } else {
            Some(Box::new(self.compile(rt, items[4])))
        };
        Ir::Try { body, catch, finally }
    }

    fn compile_let<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let binds = rt.list_to_vec(items[1]);
        self.scope.push(Vec::new());
        let mut inits = Vec::new();
        let mut i = 0;
        while i + 1 < binds.len() {
            let name = self.name(rt, binds[i]).expect("let: binding name must be a symbol");
            inits.push(self.compile(rt, binds[i + 1]));
            self.scope.last_mut().unwrap().push(name);
            i += 2;
        }
        let body: Vec<Ir> = items[2..].iter().map(|&f| self.compile(rt, f)).collect();
        self.scope.pop();
        Ir::Let(inits, Box::new(Ir::Do(body)))
    }

    fn fresh_site(&mut self) -> usize {
        let s = self.site;
        self.site += 1;
        s
    }

    fn fresh_field_site(&mut self) -> usize {
        let s = self.field_site;
        self.field_site += 1;
        s
    }

    /// If `form` is `(quote sym)`, the interned `sym`; used to lift a `.-field`
    /// name (a compile-time constant) out of its quote for the inline cache.
    fn quoted_sym<M: ValueModel>(&self, rt: &Runtime<M>, form: u64) -> Option<Sym> {
        let items = rt.list_to_vec(form);
        if items.len() == 2 && matches!(rt.decode(items[0]), Val::Sym(s) if rt.sym_name(s) == "quote") {
            if let Val::Sym(s) = rt.decode(items[1]) {
                return Some(s);
            }
        }
        None
    }
}
