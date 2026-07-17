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
    instance_site: usize,
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
    /// per-ns: `(:import …)` — simple class name -> fully-qualified class name.
    /// NAME RESOLUTION only; all class SEMANTICS live in the in-language
    /// `-jvm-registry` (see host_jvm_src).
    imports: HashMap<String, HashMap<String, String>>,
}

/// The auto-imported class names (Clojure auto-imports `java.lang.*`; this
/// dialect's finite equivalent, plus its own host types). Pure NAME table —
/// what these classes *mean* is defined by `defclass` entries in the registry.
const DEFAULT_IMPORTS: &[(&str, &str)] = &[
    ("Object", "java.lang.Object"),
    ("Runtime", "java.lang.Runtime"),
    ("String", "java.lang.String"),
    ("CharSequence", "java.lang.CharSequence"),
    ("Character", "java.lang.Character"),
    ("Number", "java.lang.Number"),
    ("Long", "java.lang.Long"),
    ("Integer", "java.lang.Integer"),
    ("Short", "java.lang.Short"),
    ("Byte", "java.lang.Byte"),
    ("Double", "java.lang.Double"),
    ("Float", "java.lang.Float"),
    ("Boolean", "java.lang.Boolean"),
    ("Math", "java.lang.Math"),
    ("System", "java.lang.System"),
    ("Class", "java.lang.Class"),
    ("Throwable", "java.lang.Throwable"),
    ("Exception", "java.lang.Exception"),
    ("RuntimeException", "java.lang.RuntimeException"),
    ("IllegalArgumentException", "java.lang.IllegalArgumentException"),
    ("IllegalStateException", "java.lang.IllegalStateException"),
    ("UnsupportedOperationException", "java.lang.UnsupportedOperationException"),
    ("IndexOutOfBoundsException", "java.lang.IndexOutOfBoundsException"),
    ("ArithmeticException", "java.lang.ArithmeticException"),
    ("NullPointerException", "java.lang.NullPointerException"),
    ("ClassCastException", "java.lang.ClassCastException"),
    ("ClassNotFoundException", "java.lang.ClassNotFoundException"),
    ("NumberFormatException", "java.lang.NumberFormatException"),
    ("Error", "java.lang.Error"),
    ("AssertionError", "java.lang.AssertionError"),
    ("StackOverflowError", "java.lang.StackOverflowError"),
    ("Symbol", "clojure.lang.Symbol"),
    ("Keyword", "clojure.lang.Keyword"),
    ("Pattern", "java.util.regex.Pattern"),
    // dialect-native host types (cljs heritage)
    ("MapEntry", "cljs.core.MapEntry"),
    ("PersistentQueue", "cljs.core.PersistentQueue"),
];

impl Compiler {
    pub fn new<M: ValueModel>(rt: &mut Runtime<M>) -> Self {
        use Prim::*;
        let mut prims = HashMap::new();
        for (name, p) in [
            ("%add", Add), ("%sub", Sub), ("%mul", Mul), ("%lt", Lt), ("%num-eq", Eq),
            // `%eq` is object IDENTITY (encoded-bits equality) — what
            // `identical?` means. `%num-eq` above is Prim::Eq, which is
            // structural `equal?`; defining `identical?` in terms of it made
            // `identical?` a synonym for `=`.
            ("%eq", Identical),
            // Keywords are INTERNED: `(keyword "a")` must return the very same
            // object as the literal `:a`, never an equal-but-distinct record.
            ("%keyword", Keyword),
            // ONE dispatch-table lookup, vs %method-types' lock+full-scan+alloc.
            ("%method-has-type?", MethodHasType),
            ("%quot", Quot), ("%rem", Rem), ("%mod", Mod), ("%div", Div), ("%str-cat", StrCat), ("%str-of", StrOf),
            ("%apply", Apply),
            // Array substrate + bitwise ops for in-language persistent structures.
            ("%make-array", MakeArray), ("%aclone", AClone), ("%alength", VectorLen),
            ("%aget", VectorRef), ("%anew", Vector),
            // Native persistent-vector ops (the trie, in Rust — see runtime `prim`).
            ("%pv-conj", PvConj), ("%pv-nth", PvNth), ("%pv-assoc", PvAssoc),
            ("%lazy-realize!", LazyRealize), ("%range-fill", RangeFill),
            ("%hamt-assoc", HamtAssoc), ("%hamt-lookup", HamtLookup), ("%hamt-without", HamtWithout),
            ("%tv-new", TvNew), ("%tv-conj!", TvConj), ("%tv-assoc!", TvAssoc),
            ("%tv-nth", TvNth), ("%tv-pop!", TvPop), ("%tv-persistent!", TvPersistent),
            ("%tam-new", TamNew), ("%tam-assoc!", TamAssoc), ("%tam-dissoc!", TamDissoc),
            ("%tam-persistent!", TamPersistent),
            ("%thm-new", ThmNew), ("%thm-assoc!", ThmAssoc), ("%thm-dissoc!", ThmDissoc),
            ("%thm-persistent!", ThmPersistent),
            ("%str-join-arr", StrJoinArr), ("%str-cmp", StrCmp), ("%pv-conj-chunk", PvConjChunk),
            ("%pv-from-array", PvFromArray), ("%apush-chunk", ApushChunk),
            ("%multifn", MultiFnNew), ("%sort-arr", SortArr),
            ("%apush", ArrPush), ("%ashift", ArrShift), ("%aclear", ArrClear),
            ("%read-string", ReadString), ("%eval", Eval), ("%macroexpand-1", MacroExpand1),
            ("%numerator", Numerator), ("%denominator", Denominator), ("%bigint?", BigIntP), ("%to-long", ToLong),
            ("%bit-and", BitAnd), ("%bit-or", BitOr), ("%bit-xor", BitXor),
            ("%bit-shl", BitShl), ("%bit-shr", BitShr), ("%bit-count", BitCount),
            ("%register-fields", RegisterFields), ("%field-by-name", FieldByName), ("%field-names", FieldNames), ("%make-record", MakeRecord), ("%hash", Hash),
            ("%first", First), ("%rest", Rest), ("%cons", Cons),
            ("record", Record), ("field", Field), ("type-of", TypeOf), ("nfields", NFields), ("throw", Throw),
            ("nil?", IsNil), ("list", List), ("%println", Println), ("%print", Print), ("gc", Gc),
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
            ("%var-flags", VarFlags), ("%ns-interns", NsInterns), ("%ns-aliases", NsAliases), ("%resolve-in-ns", ResolveInNs), ("%rand", Rand), ("%cpu-count", CpuCount), ("%amap-get", AmapGet), ("%proto-has-type?", ProtoHasType), ("%scalar-type?", ScalarType), ("%all-ns", AllNs),
            ("%symbol", SymbolOf), ("%var-arglists", VarArglists),
            // THE string-introspection primitive; clojure.string/regex build on it.
            ("%str->chars", StrChars), ("%method-types", MethodTypes),
            // Byte-level string conversions (UTF-8, signed JVM-style bytes) — the
            // substrate wire protocols (bencode/nREPL) are written on.
            ("%str->bytes", StrToBytes), ("%bytes->str", BytesToStr),
            // TCP sockets (blocking; used from real threads) — java.net models these.
            ("%tcp-listen", TcpListen), ("%tcp-accept", TcpAccept),
            ("%tcp-read", TcpRead), ("%tcp-write", TcpWrite),
            ("%tcp-close", TcpClose), ("%tcp-local-port", TcpLocalPort),
            // stderr sibling of %print (*err*'s default writer).
            ("%err-print", ErrPrint),
            // the current namespace's name symbol (frontend state, via the bridge).
            ("%current-ns", CurrentNs),
            // monotonic nanoseconds (System/nanoTime's shape) — benchmarking.
            ("%nanos", Nanos), ("%pow", Pow),
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
            instance_site: 0,
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

    /// Like `resolve_ref`, but WITHOUT the private-var access check — for `(var x)`
    /// / `#'ns/x` (var-quote), which in Clojure legitimately reaches PRIVATE vars
    /// (`@#'some.ns/private-fn` is the standard idiom to test a private).
    pub fn resolve_ref_allow_private<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym) -> Sym {
        self.resolve_global_checked(rt, s, false)
    }

/// Names `syntax-quote` must NOT namespace-qualify: this dialect's special forms
/// and the expander-level forms, which the compiler matches by BARE name. (Clojure
/// qualifies `fn`/`let`/… to clojure.core because they are macros there; here they
/// are compiler forms, so they stay bare.) `&` is in the list because it is param
/// syntax, not a value. Interop names (`.m`, `Ctor.`) are handled by shape.
const SYNTAX_QUOTE_BARE: &[&str] = &[
    // compile.rs forms
    "def", "do", "if", "quote", "set!", "try*", "-proto-method",
    // lib.rs expander forms
    "alias", "binding", "definterface", "defprotocol", "deftype", "extend-type", "fn", "fn*",
    "import", "in-ns", "instance?", "let", "let*", "loop", "loop*", "new", "ns", "refer",
    "require", "syntax-quote", "try", "use", "var", "with-redefs",
    // control forms matched by name
    "recur", "throw", "catch", "finally", "&",
];

    /// Namespace-qualify a symbol the way `syntax-quote` does (Clojure macro
    /// hygiene): a symbol that resolves to a KNOWN var (own def, refer, or
    /// clojure.core) becomes its fully-qualified name, so a macro's references to
    /// its own/ core helpers resolve correctly at the EXPANSION site regardless of
    /// the caller's namespace. Special forms (`let`/`if`/`fn`/…), prims, `&`, and
    /// genuinely unknown symbols are left BARE (the compiler matches those by bare
    /// name; qualifying them would break dispatch).
    pub fn qualify_for_syntax_quote<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym) -> Sym {
        if self.prims.contains_key(&s) {
            return s;
        }
        let name = rt.sym_name(s);
        if name.len() > 1 && name.contains('/') {
            // already qualified / alias-prefixed -> resolve normally
            return self.resolve_global(rt, s);
        }
        let ns = &self.ns.current;
        let q = if self.ns.ns_defs.get(ns).is_some_and(|d| d.contains(name)) {
            rt.intern(&format!("{ns}/{name}"))
        } else if let Some(fq) = self.ns.refers.get(ns).and_then(|r| r.get(name)) {
            rt.intern(fq)
        } else if self.ns.ns_defs.get("clojure.core").is_some_and(|d| d.contains(name)) {
            rt.intern(&format!("clojure.core/{name}"))
        } else if Self::SYNTAX_QUOTE_BARE.contains(&name)
            || name.starts_with('.')
            || name.ends_with('.')
        {
            // A SPECIAL FORM (or interop syntax) — the compiler matches these by
            // bare name, so qualifying them would break dispatch. Clojure leaves
            // its specials bare here for the same reason.
            return s;
        } else if let Some(fqn) = self.resolve_class(name) {
            // A class name resolves to the fully-qualified class, as in Clojure:
            // `` `Exception `` is java.lang.Exception, so a macro's catch clause
            // means the same class at the expansion site.
            rt.intern(&fqn)
        } else {
            // Genuinely unknown: qualify with the CURRENT namespace. This is
            // Clojure's rule and it is not cosmetic — it is what makes a
            // syntax-quoted symbol name the same thing at every expansion site.
            // Leaving it bare silently produced a DIFFERENT symbol: meander
            // registers its `with` parser under `` `with ``, expecting
            // meander.syntax.epsilon/with, and looked it up by that name — the
            // bare key never matched, so `with` was not recognized as syntax at
            // all ("Unbound reference %x").
            rt.intern(&format!("{ns}/{name}"))
        };
        // Protocol methods (`-conj`, `-seq`, …) dispatch by NAME — a qualified
        // reference won't resolve as a method, so keep them bare.
        if rt.is_method_name(q) {
            return s;
        }
        q
    }

    /// `(ns-resolve ns sym)` — resolve `name` to a fully-qualified var name AS
    /// SEEN FROM the namespace `ns`, or None if no such var exists.
    ///
    /// This is the RUNTIME twin of the compile-time rewrite that `resolve` on a
    /// literal gets. It cannot use that rewrite: the symbol here is a runtime
    /// VALUE (meander hands `resolve-symbol` symbols it read out of a pattern),
    /// so there is no literal to rewrite and every lookup answered nil.
    /// Resolution order matches the compiler's: own defs -> :refer -> clojure.core.
    /// A qualified `alias/name` resolves the alias against `ns` first.
    pub fn resolve_var_in_ns(&self, ns: &str, name: &str) -> Option<String> {
        let defines = |n: &str, v: &str| self.ns.ns_defs.get(n).is_some_and(|d| d.contains(v));
        if name.len() > 1 && name.contains('/') {
            let (pre, base) = name.split_once('/')?;
            // `pre` may be an alias in `ns`, or already a real namespace name.
            let real = self
                .ns
                .aliases
                .get(ns)
                .and_then(|m| m.get(pre))
                .cloned()
                .unwrap_or_else(|| pre.to_string());
            return defines(&real, base).then(|| format!("{real}/{base}"));
        }
        if defines(ns, name) {
            return Some(format!("{ns}/{name}"));
        }
        if let Some(fq) = self.ns.refers.get(ns).and_then(|r| r.get(name)) {
            return Some(fq.clone());
        }
        defines("clojure.core", name).then(|| format!("clojure.core/{name}"))
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

    /// `(:import (java.util UUID))` — the bare simple name now resolves to the
    /// fully-qualified class name in interop positions.
    pub fn add_import(&mut self, simple: &str, fqn: &str) {
        let ns = self.ns.current.clone();
        self.ns.imports.entry(ns).or_default().insert(simple.to_string(), fqn.to_string());
    }

    /// Resolve a bare class-ish name to a fully-qualified class name: explicit
    /// per-ns imports first, then the auto-import table. `None` = not a known
    /// class name (a deftype / dialect record tag).
    pub fn resolve_class(&self, simple: &str) -> Option<String> {
        if let Some(f) = self.ns.imports.get(&self.ns.current).and_then(|m| m.get(simple)) {
            return Some(f.clone());
        }
        DEFAULT_IMPORTS.iter().find(|(s, _)| *s == simple).map(|(_, f)| f.to_string())
    }

    /// Is `alias` a registered namespace alias in the current ns? (A capitalized
    /// alias must win over class-name interpretation of `Alias/member`.)
    pub fn has_alias(&self, alias: &str) -> bool {
        self.ns.aliases.get(&self.ns.current).is_some_and(|m| m.contains_key(alias))
    }

    /// Every `alias -> real` pair registered in the namespace `ns`, sorted by
    /// alias so the reflected table has a stable order. (`ns-aliases` reflects an
    /// ARBITRARY namespace, so this cannot read `self.ns.current`.)
    pub fn aliases_of(&self, ns: &str) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .ns
            .aliases
            .get(ns)
            .map(|m| m.iter().map(|(a, r)| (a.clone(), r.clone())).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    /// The namespace an alias points to in the current ns (for `::alias/kw`).
    pub fn alias_target(&self, alias: &str) -> Option<String> {
        self.ns.aliases.get(&self.ns.current).and_then(|m| m.get(alias)).cloned()
    }

    /// `[foo :refer [x]]` — bare `x` now resolves to the fully-qualified `foo/x`.
    pub fn add_refer(&mut self, short: &str, fq: &str) {
        let ns = self.ns.current.clone();
        self.ns.refers.entry(ns).or_default().insert(short.to_string(), fq.to_string());
    }

    /// `(use 'foo)` / `(:use foo)` — refer EVERY public name of `foo`: its defined
    /// vars AND its protocol/interface method names (which aren't vars, so a plain
    /// var-refer would miss them).
    pub fn refer_all<M: ValueModel>(&mut self, rt: &Runtime<M>, from: &str) {
        let prefix = format!("{from}/");
        let method_refers: Vec<(String, String)> = self
            .methods
            .iter()
            .filter_map(|&m| {
                let name = rt.sym_name(m).to_string();
                name.strip_prefix(&prefix).map(|short| (short.to_string(), name.clone()))
            })
            .collect();
        for (short, fq) in method_refers {
            self.add_refer(&short, &fq);
        }
        if let Some(defs) = self.ns.ns_defs.get(from).cloned() {
            for name in defs {
                self.add_refer(&name, &format!("{from}/{name}"));
            }
        }
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
        self.resolve_global_checked(rt, s, true)
    }
    fn resolve_global_checked<M: ValueModel>(&self, rt: &Runtime<M>, s: Sym, check_private: bool) -> Sym {
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
                    // `cljs.core/x` resolves to `clojure.core/x` — this dialect is
                    // JVM-free like ClojureScript, so cljs-targeted library code
                    // referring to cljs.core builtins finds our core equivalents.
                    let real = if real == "cljs.core" { "clojure.core" } else { real };
                    let q = rt.intern(&format!("{real}/{right}"));
                    // A `^:private` / `defn-` var is only accessible within its ns
                    // (var-quote `#'ns/x` bypasses this, like Clojure).
                    if check_private && real != self.ns.current && self.private.contains(&q) {
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

    /// `(f 'a 'b …)` — a call to a (clojure.core) runtime fn with symbol
    /// constants for arguments; how compile-time name resolution hands a class
    /// reference to the in-language JVM layer.
    fn jvm_call<M: ValueModel>(&self, rt: &mut Runtime<M>, f: &str, syms: &[Sym]) -> Ir {
        let g = Ir::Global(rt.intern(f));
        let args = syms.iter().map(|&x| self.sym_const(rt, x)).collect();
        Ir::Call(Box::new(g), args)
    }

    /// Compile a (non-local) symbol reference: a `^:dynamic` var reads the
    /// thread-local binding stack (`%dyn-get`); any other var is a plain global.
    fn global_ref<M: ValueModel>(&self, rt: &mut Runtime<M>, s: Sym) -> Ir {
        // `*ns*` reflects the LIVE compiler namespace (read-only; rebinding it
        // is not supported — switch with `ns`/`in-ns`).
        if rt.sym_name(s) == "*ns*" {
            return self.jvm_call(rt, "clojure.core/-ns-object", &[]);
        }
        // `Math/PI`-style STATIC reference in value position: a capitalized,
        // non-alias left segment that is a dotted class name or an imported
        // simple name reads the class's static member through the JVM layer.
        // Also `ns/Class.MEMBER` (cljs style: `cljs.core/PersistentQueue.EMPTY`)
        // — a dotted, capitalized RIGHT side is class + member under the ns.
        {
            let nm = rt.sym_name(s).to_string();
            if let Some(slash) = nm.find('/') {
                let (left, right) = (&nm[..slash], &nm[slash + 1..]);
                let left_caps = left
                    .rsplit('.')
                    .next()
                    .and_then(|x| x.chars().next())
                    .is_some_and(|c| c.is_ascii_uppercase());
                if !right.is_empty() && left_caps && !self.has_alias(left) {
                    let fqn = if left.contains('.') {
                        Some(left.to_string())
                    } else {
                        self.resolve_class(left)
                    };
                    if let Some(fqn) = fqn {
                        let fq = rt.intern(&fqn);
                        let mem = rt.intern(right);
                        return self.jvm_call(rt, "clojure.core/-jvm-static-member", &[fq, mem]);
                    }
                }
                if !left_caps && right.contains('.') {
                    if let Some(dot) = right.rfind('.') {
                        let (cls, mem) = (&right[..dot], &right[dot + 1..]);
                        if cls.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                            && !mem.is_empty()
                        {
                            let fq = rt.intern(&format!("{left}.{cls}"));
                            let mem = rt.intern(mem);
                            return self.jvm_call(rt, "clojure.core/-jvm-static-member", &[fq, mem]);
                        }
                    }
                }
            }
        }
        // A bare dotted class reference in VALUE position (`clojure.lang.
        // IPersistentVector` as a dispatch value — real core.match does this).
        // Prefer a var `a.b/C` (a protocol/deftype defined in ns `a.b`); else
        // hand the name to the JVM layer, which yields a `Class` record for
        // registered classes, a static member for `pkg.Class.MEMBER` js-style
        // refs, or the bare tag symbol for unknown names (legacy behavior).
        {
            let nm = rt.sym_name(s).to_string();
            let classlike = !nm.contains('/')
                && nm.contains('.')
                && nm
                    .rsplit('.')
                    .next()
                    .and_then(|seg| seg.chars().next())
                    .is_some_and(|c| c.is_ascii_uppercase());
            if classlike {
                if let Some(pos) = nm.rfind('.') {
                    let dotted_var = rt.intern(&format!("{}/{}", &nm[..pos], &nm[pos + 1..]));
                    if rt.global_defined(dotted_var) {
                        return Ir::Global(dotted_var);
                    }
                }
                let simple = nm.rsplit('.').next().unwrap_or(&nm).to_string();
                let parent = nm[..nm.len() - simple.len()].trim_end_matches('.').to_string();
                let fq = rt.intern(&nm);
                let sim = rt.intern(&simple);
                let par = rt.intern(&parent);
                return self.jvm_call(rt, "clojure.core/-jvm-class-value", &[fq, sim, par]);
            }
        }
        // A prim used in VALUE position (`(map nil? xs)`, `{:a nil?}`) has no var to
        // read — synthesize a wrapper closure so it's a first-class function. In
        // CALL position the prim still inlines (checked before this path).
        if let Some(&p) = self.prims.get(&s) {
            // `-to-list` is resolved (not inlined) so `list`'s wrapper can call
            // it; it is late-bound like any global, so bootstrap order is fine.
            let to_list = self.resolve_global(rt, rt.intern("-to-list"));
            if let Some(wrap) = prim_value_wrapper(p, to_list) {
                return wrap;
            }
        }
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
            let id = rt.alloc_record(vsym, &[params]);
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
                // `(%dispatch method args…)` — an UNCONDITIONAL protocol-dispatch
                // site. `defprotocol` defs each method as a first-class fn whose
                // body is this form, so `(map prepend …)` / `(reduce val-at …)`
                // work; a plain `(method …)` call may route through that var, and
                // the wrapper must still dispatch rather than call itself.
                "%instance-check" => {
                    // items: [%instance-check (quote -instance-val) proto-ref arg]
                    let ivq = rt.list_to_vec(items[1]);
                    let iv = self
                        .name(rt, ivq[1])
                        .expect("%instance-check: -instance-val sym");
                    let iv = self.resolve_global(rt, iv);
                    let proto = Box::new(self.compile(rt, items[2]));
                    let arg = Box::new(self.compile(rt, items[3]));
                    let site = self.fresh_instance_site();
                    return Ir::InstanceCheck { site, iv, proto, arg };
                }
                "%dispatch" => {
                    let m_raw = self.name(rt, items[1]).expect("%dispatch: method name");
                    // A DOT-name (`.charAt` — a host instance method) is its own
                    // dispatch key, un-namespaced: JVM method names aren't
                    // namespaced, and registrations from any ns must share it.
                    let method = if rt.sym_name(m_raw).starts_with('.') {
                        m_raw
                    } else {
                        self.resolve_global(rt, m_raw)
                    };
                    let site = self.fresh_site();
                    let args = items[2..].iter().map(|&f| self.compile(rt, f)).collect();
                    return Ir::Dispatch { site, method, args };
                }
                "-proto-method" => {
                    let m_raw = self.name(rt, items[1]).expect("defmethod: method name");
                    let ty = self.name(rt, items[2]).expect("defmethod: type name");
                    // A protocol method is a namespace-qualified var (as in Clojure).
                    // The declaration from `defprotocol` (ty = the -protocol-default
                    // sentinel) DEFINES it in the current ns; an `extend-type` impl
                    // RESOLVES it (own -> refer -> auto-referred clojure.core). The
                    // TYPE tag stays bare — it's a `type-of` record tag, not a var.
                    // A DOT-name (host instance method, from `defclass`) stays a
                    // bare un-namespaced key (see `%dispatch`).
                    let m = if rt.sym_name(m_raw).starts_with('.') {
                        m_raw
                    } else if rt.sym_name(ty) == "-protocol-default" {
                        self.def_name(rt, m_raw)
                    } else {
                        self.resolve_global(rt, m_raw)
                    };
                    self.methods.insert(m);
                    let imp = self.compile(rt, items[3]);
                    return Ir::DefMethod { name: m, ty, imp: Box::new(imp) };
                }
                _ => {
                    // Inline core arithmetic/comparison to prims (Clojure's `:inline`):
                    // `(+ a b)` compiles straight to `Ir::Prim(Add,…)`, skipping the
                    // variadic operator fn's rest-arg list + seq fold — the single
                    // biggest cost in numeric code. Fires ONLY when the head is not a
                    // local and resolves to the canonical `clojure.core` var, so a
                    // local binding or a user redefinition still calls the real fn.
                    if self.resolve_local(hs).is_none() {
                        let resolved = self.resolve_global(rt, hs);
                        let rname = rt.sym_name(resolved).to_string();
                        if let Some(op) = rname.strip_prefix("clojure.core/") {
                            if let Some(ir) = self.try_inline_op(rt, op, &items[1..]) {
                                return ir;
                            }
                        }
                    }
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
                    // Dispatch a protocol method. cljs-style protocol methods are
                    // dash-prefixed (`-nth`, `-meta`) and ALWAYS dispatch. A non-dash
                    // name (e.g. `nth`/`get`/`conj`) that a JVM library registered as an
                    // interface method AND that is also a real fn var means the code wants
                    // the fn (its seq fallbacks + internal `-nth` dispatch), so let it fall
                    // through to the ordinary call.
                    let mname = rt.sym_name(method);
                    let is_dash = mname.rsplit('/').next().unwrap_or(mname).starts_with('-');
                    let dispatch = self.methods.contains(&method)
                        && (is_dash || !rt.global_defined(method));
                    if dispatch {
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

    /// The `:inline` table: a core operator applied at a fixed low arity lowers to
    /// the matching `%`-prim directly. `None` = not inlinable at this arity (0/1-arg
    /// `+`, negation, chained `<`, `=`, `/`, …) — the caller falls back to the fn.
    /// Semantics are byte-identical to the operator fns in `core.clj`: the prims
    /// carry the numeric tower (int/ratio/bigint promotion), so results match.
    fn try_inline_op<M: ValueModel>(
        &mut self,
        rt: &mut Runtime<M>,
        op: &str,
        arg_forms: &[u64],
    ) -> Option<Ir> {
        use Prim::{Add, Lt, Mul, Sub};
        let argc = arg_forms.len();
        let konst = |rt: &mut Runtime<M>, v: Val| {
            let h = rt.encode(v);
            Ir::Const(rt.intern_const(h))
        };
        match op {
            // n-ary fold, 2+ args: (+ a b c) -> (add (add a b) c). 0/1-arg keep the fn.
            "+" | "-" | "*" if argc >= 2 => {
                let prim = match op {
                    "+" => Add,
                    "-" => Sub,
                    _ => Mul,
                };
                let mut acc = self.compile(rt, arg_forms[0]);
                for &a in &arg_forms[1..] {
                    let r = self.compile(rt, a);
                    acc = Ir::Prim(prim, vec![acc, r]);
                }
                Some(acc)
            }
            // Two-arg comparisons; chained (>2) forms keep the fn.
            "<" | ">" if argc == 2 => {
                let a = self.compile(rt, arg_forms[0]);
                let b = self.compile(rt, arg_forms[1]);
                Some(if op == "<" {
                    Ir::Prim(Lt, vec![a, b])
                } else {
                    Ir::Prim(Lt, vec![b, a])
                })
            }
            "<=" | ">=" if argc == 2 => {
                // x <= y  ==  not (y < x);  x >= y  ==  not (x < y).
                let a = self.compile(rt, arg_forms[0]);
                let b = self.compile(rt, arg_forms[1]);
                let (lo, hi) = if op == "<=" { (b, a) } else { (a, b) };
                let lt = Ir::Prim(Lt, vec![lo, hi]);
                let f = konst(rt, Val::Bool(false));
                let t = konst(rt, Val::Bool(true));
                Some(Ir::If(Box::new(lt), Box::new(f), Box::new(t)))
            }
            "inc" | "dec" if argc == 1 => {
                let a = self.compile(rt, arg_forms[0]);
                let one = konst(rt, Val::Int(1));
                let prim = if op == "inc" { Add } else { Sub };
                Some(Ir::Prim(prim, vec![a, one]))
            }
            _ => None,
        }
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
        // Placeholders; the `flatten` pass computes the real nslots/captures.
        Ir::Lambda { nparams, variadic, nslots: 0, captures: Vec::new(), body: Arc::new(Ir::Do(body)) }
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
        Ir::Try { body, catch, finally, cslot: 0, site: microlang::ir::fresh_try_site() }
    }

    fn compile_let<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let binds = rt.list_to_vec(items[1]);
        self.scope.push(Vec::new());
        let mut inits = Vec::new();
        let mut i = 0;
        while i + 1 < binds.len() {
            // Name the offending form: by this point `destructure` has already
            // rewritten every pattern it understands into plain symbols, so a
            // non-symbol here means an UNSUPPORTED binding shape — and "must be
            // a symbol" alone gives no way to find out which one.
            let name = self.name(rt, binds[i]).unwrap_or_else(|| {
                panic!("let: unsupported binding form (not a symbol): {}", rt.print(binds[i]))
            });
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

    fn fresh_instance_site(&mut self) -> usize {
        let s = self.instance_site;
        self.instance_site += 1;
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

/// A first-class wrapper closure for a prim referenced in VALUE position, so
/// `nil?`/`list`/etc. can be passed around like any function. `None` for prims
/// that are only ever internal (`%`-prefixed) and never used as bare values.
fn prim_value_wrapper(p: Prim, to_list: Sym) -> Option<Ir> {
    // body references the wrapper's params via Local{up:0, idx:i}
    let local = |i: u16| Ir::Local { up: 0, idx: i };
    let fixed = |p: Prim, n: u16| {
        let args = (0..n).map(local).collect();
        // Already flat: params are the only slots, nothing captured.
        Ir::Lambda {
            nparams: n as usize,
            variadic: false,
            nslots: n,
            captures: Vec::new(),
            body: Arc::new(Ir::Prim(p, args)),
        }
    };
    match p {
        // `(fn [& xs] (-to-list xs))`. NOT `(fn [& xs] xs)`: a rest arg is no
        // longer necessarily a list. `apply` hands the callee the applied seq
        // itself (Clojure's structure sharing), so returning it raw made
        // `(apply list (range 3))` answer a ChunkedCons where Clojure gives a
        // PersistentList. `list` must BUILD one, as PersistentList/create does.
        Prim::List => Some(Ir::Lambda {
            nparams: 0,
            variadic: true,
            nslots: 1,
            captures: Vec::new(),
            body: Arc::new(Ir::Call(
                Box::new(Ir::Global(to_list)),
                vec![local(0)],
            )),
        }),
        Prim::IsNil | Prim::TypeOf | Prim::Throw | Prim::Hash | Prim::NFields => Some(fixed(p, 1)),
        Prim::Field => Some(fixed(p, 2)),
        _ => None,
    }
}
