//! Clojure -> `Ir`, DIRECTLY. This frontend does NOT ride the toolkit's s-expr
//! `analyze`/special-forms — it compiles fully-expanded Clojure forms to the
//! toolkit's neutral `Ir` itself, owning name resolution, its own core forms
//! (`fn`/`let`/`if`/`do`/`def`/`quote`/`set!`/`defmethod`), prim mapping, and
//! protocol dispatch. The toolkit's job starts at `Ir`.

use std::collections::HashMap;
use std::sync::Arc;

use microlang::ir::{Ir, Prim};
use microlang::value::Sym;
use microlang::{Obj, Runtime, Val, ValueModel};

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
    /// Resolved syms declared `^:dynamic`: a reference to one reads the thread-local
    /// binding stack (`%dyn-get`) rather than the plain global.
    dynamics: std::collections::HashSet<Sym>,
    /// Resolved syms declared `^:private` / `defn-`: a qualified reference from a
    /// DIFFERENT namespace is a compile error.
    private: std::collections::HashSet<Sym>,
    /// Namespace state (frontend-owned; the toolkit's globals are a flat `Sym`
    /// pool, so a user-ns var `foo/x` is simply the interned sym `"foo/x"`).
    ns: NsState,
    /// Directories searched to turn a required namespace `foo.bar` into a source
    /// file `foo/bar.clj` (or `.cljc`). `require` reads and loads it on demand.
    load_paths: Vec<std::path::PathBuf>,
    /// Namespaces already loaded (embedded or from a file), so `require` loads
    /// each at most once and cycles terminate.
    loaded: std::collections::HashSet<String>,
}

/// Namespace resolution state. EVERY var is namespace-qualified: `clojure.core` +
/// the ported cljs types load into the `clojure.core` namespace, then user code
/// runs in `user` (or whatever `ns`/`in-ns` sets). A bare reference resolves
/// own-def → refer → auto-referred `clojure.core` → own-ns (forward ref). The only
/// bare symbols are frontend prims (`%add`) and protocol-method dispatch keys.
#[derive(Default)]
struct NsState {
    current: String,
    /// per-ns: the set of names `def`'d there (also the auto-refer source for
    /// `clojure.core`).
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
            ("%quot", Quot), ("%rem", Rem), ("%mod", Mod), ("%div", Div), ("%str-cat", StrCat), ("%str-of", StrOf),
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
            // Dynamic vars (`binding` desugars to these; refs/set! emit DynGet/DynSet).
            ("%dyn-mark", DynMark), ("%dyn-bind", DynBind), ("%dyn-unwind", DynUnwind),
            // First-class vars: read/write a global by symbol (the Var handle's field).
            ("%global-get", GlobalGet), ("%global-set", GlobalSet), ("%global-bound?", GlobalBound),
            // Split a (possibly qualified) symbol for var reflection (name/namespace).
            ("%sym-name", SymName), ("%sym-ns", SymNs),
            // Var/namespace registry reflection (metadata flags + ns enumeration).
            ("%var-flags", VarFlags), ("%ns-interns", NsInterns), ("%all-ns", AllNs),
            ("%symbol", SymbolOf), ("%var-arglists", VarArglists),
            // THE string-introspection primitive; clojure.string/regex build on it.
            ("%str->chars", StrChars),
            // Existing low-level prims surfaced for the string library (char codes
            // for case mapping, raw length) — not new primitives.
            ("%char-code", CharToInt), ("%char-of", IntToChar), ("%str-len", StrLen),
        ] {
            prims.insert(rt.intern(name), p);
        }
        Compiler {
            scope: Vec::new(),
            dynamics: std::collections::HashSet::new(),
            private: std::collections::HashSet::new(),
            methods: std::collections::HashSet::new(),
            site: 0,
            field_site: 0,
            prims,
            // core.clj + the ported cljs types load into the `clojure.core` ns.
            ns: NsState { current: "clojure.core".to_string(), ..NsState::default() },
            load_paths: Vec::new(),
            loaded: std::collections::HashSet::new(),
        }
    }

    /// Directories `require` searches for a namespace's source file.
    pub fn set_load_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        self.load_paths = paths;
    }

    /// The load-path directories (for the frontend's file loader).
    pub fn load_paths(&self) -> &[std::path::PathBuf] {
        &self.load_paths
    }

    /// Mark a namespace loaded (embedded prelude, or a file just loaded).
    pub fn mark_loaded(&mut self, ns: &str) {
        self.loaded.insert(ns.to_string());
    }

    /// Has this namespace already been loaded?
    pub fn is_loaded(&self, ns: &str) -> bool {
        self.loaded.contains(ns)
    }

    /// The namespace currently being compiled (saved/restored around a file load).
    pub fn current_ns(&self) -> &str {
        &self.ns.current
    }

    /// Called by `run` once clojure.core + the cljs types are loaded: subsequent
    /// (user) code qualifies its defs into `user` (or whatever `ns`/`in-ns` sets).
    pub fn end_core_load(&mut self) {
        self.ns.current = "user".to_string();
    }

    /// Resolve a symbol the way a reference would, per the current namespace —
    /// exposed so the frontend can locate a macro's fn global (it may have been
    /// defined into the current ns).
    pub fn resolve_ref<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym) -> Sym {
        self.resolve_global(rt, s)
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
                None => self.global_ref(rt, s),
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
        // Frontend prims (`%add`, `list`, …) are never namespace vars.
        if self.prims.contains_key(&s) {
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
                    let q = rt.intern(&format!("{real}/{right}"));
                    // A `^:private` / `defn-` var is only accessible within its ns.
                    if real != self.ns.current && self.private.contains(&q) {
                        panic!("var {real}/{right} is private (declared with defn-/^:private)");
                    }
                    return q;
                }
            }
        }
        let ns = &self.ns.current;
        if self.ns.ns_defs.get(ns).is_some_and(|d| d.contains(name)) {
            return rt.intern(&format!("{ns}/{name}")); // own def (shadows core)
        }
        if let Some(fq) = self.ns.refers.get(ns).and_then(|r| r.get(name)) {
            return rt.intern(fq);
        }
        // Auto-referred clojure.core: every namespace sees its vars.
        if self.ns.ns_defs.get("clojure.core").is_some_and(|d| d.contains(name)) {
            return rt.intern(&format!("clojure.core/{name}"));
        }
        // Unknown bare name: assume the current ns (forward ref to a var defined
        // later here, or an as-yet-unbound var).
        rt.intern(&format!("{ns}/{name}"))
    }

    /// Compile a (non-local) symbol reference: a `^:dynamic` var reads the
    /// thread-local binding stack (`%dyn-get`); any other var is a plain global.
    fn global_ref<M: ValueModel>(&self, rt: &mut Runtime<M>, s: Sym) -> Ir {
        let r = self.resolve_global(rt, s);
        // A `^:dynamic` var reads the thread-local binding stack (keyed by its
        // resolved, qualified sym), falling back to the root global if unbound.
        if self.dynamics.contains(&r) {
            return Ir::Prim(Prim::DynGet, vec![self.sym_const(rt, r)]);
        }
        Ir::Global(r)
    }

    /// Peel `(-dynamic-meta …)` / `(-private-meta …)` wrappers off a `def` name
    /// (from `^:dynamic` / `^:private` / `defn-`), returning the bare name form and
    /// the flags. Handles either order / both.
    fn unwrap_def_meta<M: ValueModel>(&self, rt: &Runtime<M>, mut form: u64) -> (u64, bool, bool) {
        let (mut dynamic, mut private) = (false, false);
        while rt.as_cons(form).is_some() {
            let items = rt.list_to_vec(form);
            if items.len() != 2 {
                break;
            }
            match rt.decode(items[0]) {
                Val::Sym(s) if rt.sym_name(s) == "-dynamic-meta" => {
                    dynamic = true;
                    form = items[1];
                }
                Val::Sym(s) if rt.sym_name(s) == "-private-meta" => {
                    private = true;
                    form = items[1];
                }
                _ => break,
            }
        }
        (form, dynamic, private)
    }

    /// Extract `:arglists` from a `(fn …)` init form (single-arity), building a
    /// `([params])` datum captured into the var registry at compile time. Returns
    /// `None` for non-fn inits or multi-arity fns (whose arglists our basic `defn`
    /// doesn't produce anyway).
    fn fn_arglists<M: ValueModel>(&self, rt: &mut Runtime<M>, init: u64) -> Option<u64> {
        rt.as_cons(init)?;
        let items = rt.list_to_vec(init);
        let is_fn = items
            .first()
            .is_some_and(|&h| matches!(rt.decode(h), Val::Sym(s) if matches!(rt.sym_name(s), "fn" | "fn*")));
        if !is_fn {
            return None;
        }
        let mut i = 1;
        // Skip an optional self-name (`(fn* name params …)`).
        if items.get(i).is_some_and(|&x| matches!(rt.decode(x), Val::Sym(_))) {
            i += 1;
        }
        let params = *items.get(i)?;
        // Multi-arity `(fn ([a] …) ([a b] …))`: the item is a list whose head is
        // itself a (param) list — skip (single-arity only).
        if let Some((h, _)) = rt.as_cons(params) {
            if rt.as_cons(h).is_some() {
                return None;
            }
        }
        let v = self.vector_of(rt, params);
        Some(rt.vec_to_list(&[v]))
    }

    /// Present a param binder as a `Vector` record (so `:arglists` prints `[a b]`).
    /// A post-expansion param LIST is wrapped; an already-vector value is kept.
    fn vector_of<M: ValueModel>(&self, rt: &mut Runtime<M>, params: u64) -> u64 {
        if rt.as_cons(params).is_some() {
            let vsym = rt.intern("Vector");
            let id = rt.alloc(Obj::Record { type_id: vsym, fields: vec![params] });
            rt.encode(Val::Ref(id))
        } else {
            params
        }
    }

    /// A `Const` Ir holding the symbol `s` as a value (for the `%dyn-*` prims,
    /// which take the var's name as a runtime argument).
    fn sym_const<M: ValueModel>(&self, rt: &mut Runtime<M>, s: Sym) -> Ir {
        let v = rt.encode(Val::Sym(s));
        Ir::Const(rt.intern_const(v))
    }

    /// Resolve a DEFINED name to the interned sym `Ir::Def` should write, and record
    /// it so later bare references in this ns resolve to it.
    fn def_name<M: ValueModel>(&mut self, rt: &Runtime<M>, raw: Sym) -> Sym {
        // A name containing `/` is explicitly qualified — EXCEPT the lone `/`
        // symbol (division), whose name simply is "/".
        if rt.sym_name(raw).contains('/') && rt.sym_name(raw) != "/" {
            return self.resolve_global(rt, raw);
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
                    // Peel reader-lowered `^:dynamic` / `^:private` off the name.
                    let (nameform, dynamic, private) = self.unwrap_def_meta(rt, items[1]);
                    let raw = self.name(rt, nameform).expect("def: name must be a symbol");
                    // Resolve+record the name FIRST so a self-reference inside the
                    // init (e.g. `(def x (fn [] x))`) resolves to this same var.
                    let n = self.def_name(rt, raw);
                    let mut flags = 0u8;
                    if dynamic {
                        // Key the binding stack by the var's resolved, qualified sym.
                        self.dynamics.insert(n);
                        flags |= microlang::runtime::VAR_DYNAMIC;
                    }
                    if private {
                        self.private.insert(n);
                        flags |= microlang::runtime::VAR_PRIVATE;
                    }
                    // Record the var in the runtime registry (for ns enumeration and
                    // `(meta #'x)` flags).
                    rt.register_var(n, &self.ns.current, flags);
                    // Capture :arglists from a `(fn …)` init (compile-time, so it
                    // needs no bootstrap-order-sensitive emitted code).
                    if items.len() > 2 {
                        if let Some(al) = self.fn_arglists(rt, items[2]) {
                            rt.set_var_arglists(n, al);
                        }
                    }
                    if items.len() > 2 {
                        let init = Box::new(self.compile(rt, items[2]));
                        return Ir::Def { name: n, init };
                    }
                    // value-less `(def x)` / `(declare x)`: the var is INTERNED
                    // (so `#'x` resolves and `x` is a forward reference) but its
                    // root stays UNBOUND — a deref throws "Unbound" until set. The
                    // name was recorded by `def_name`; emit no store, just nil.
                    let nil = rt.encode(Val::Nil);
                    return Ir::Const(rt.intern_const(nil));
                }
                "fn" | "fn*" => return self.compile_fn(rt, &items),
                "let" | "let*" => return self.compile_let(rt, &items),
                "try*" => return self.compile_try(rt, &items),
                "set!" => {
                    let n = self.name(rt, items[1]).expect("set!: target must be a symbol");
                    let val = Box::new(self.compile(rt, items[2]));
                    return match self.resolve_local(n) {
                        Some((up, idx)) => Ir::SetLocal { up, idx, val },
                        None => {
                            let r = self.resolve_global(rt, n);
                            if self.dynamics.contains(&r) {
                                // `set!` on a dynamic var mutates the current binding.
                                Ir::Prim(Prim::DynSet, vec![self.sym_const(rt, r), *val])
                            } else {
                                Ir::SetGlobal { name: r, val }
                            }
                        }
                    };
                }
                "-proto-method" => {
                    let m_raw = self.name(rt, items[1]).expect("defmethod: method name");
                    let ty = self.name(rt, items[2]).expect("defmethod: type name");
                    // A protocol method is a namespace-qualified var (as in Clojure).
                    // The declaration from `defprotocol` (ty = the -protocol-default
                    // sentinel) DEFINES it in the current ns; an `extend-type` impl
                    // RESOLVES it (own -> refer -> auto-referred clojure.core). The
                    // TYPE tag stays bare — it's a `type-of` record tag, not a var.
                    let m = if rt.sym_name(ty) == "-protocol-default" {
                        self.def_name(rt, m_raw)
                    } else {
                        self.resolve_global(rt, m_raw)
                    };
                    self.methods.insert(m);
                    let imp = self.compile(rt, items[3]);
                    return Ir::DefMethod { name: m, ty, imp: Box::new(imp) };
                }
                _ => {
                    // A binding (local or a def'd var) shadows a prim; a prim is
                    // otherwise its default. The var check uses the RESOLVED sym.
                    let shadowed =
                        self.resolve_local(hs).is_some() || rt.global_defined(self.resolve_global(rt, hs));
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
                    let method = self.resolve_global(rt, hs);
                    if self.methods.contains(&method) {
                        let site = self.fresh_site();
                        let args = items[1..].iter().map(|&f| self.compile(rt, f)).collect();
                        return Ir::Dispatch { site, method, args };
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
