//! A mini-Clojure frontend, built entirely on the microlang toolkit's PUBLIC API.
//!
//! The toolkit deliberately has NO macro system and NO Clojure data types — both
//! are frontend policy. So this frontend brings:
//!   * a Clojure reader (`reader`): `[] {} #{} :kw "str" \c '` -> toolkit values,
//!   * its own PROCEDURAL macro expander (`defmacro` + `expand`),
//!   * Clojure's data types as list-backed tagged `Record`s (Vector/Map/Set/
//!     Keyword) with `type-of`-driven dispatch, and
//!   * a `clojure.core` prelude — seqs, collections, HOFs — written in the
//!     language on top of the toolkit's cons/record/vector primitives.
//!
//! Collection LITERALS in code position are desugared by `expand` to constructor
//! calls (so `[a b]` evaluates its elements); under `quote` they stay literal.
//! Binding vectors (`fn`/`let`) are converted back to the toolkit's list form.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use microlang::runtime::ObjView;
use microlang::value::Sym;
use microlang::{CodeSpace, EvalBridge, Obj, Runtime, Val, ValueModel};

/// Monotonic counter for auto-gensym (`foo#`). Deterministic (no RNG/clock).
static GENSYM: AtomicU64 = AtomicU64::new(0);

mod compile;
mod data;
pub mod deps;
mod reader;
mod roots;

use roots::RootVec;

/// The embedded prelude — REAL `.clj` files under `src/clj/`, compiled into
/// the binary with `include_str!` so the frontend stays self-contained while
/// the sources stay ordinary, editable Clojure files.
mod sources {
    pub const CORE: &str = include_str!("clj/core.clj");
    pub const CLJS_TYPES: &str = include_str!("clj/cljs_types.clj");
    pub const HOST_JVM: &str = include_str!("clj/host_jvm.clj");
    pub const HOST_IO: &str = include_str!("clj/host_io.clj");
    pub const CLOJURE_STRING: &str = include_str!("clj/clojure/string.clj");
    pub const CLOJURE_SET: &str = include_str!("clj/clojure/set.clj");
    pub const CLOJURE_WALK: &str = include_str!("clj/clojure/walk.clj");
    pub const CLOJURE_PPRINT: &str = include_str!("clj/clojure/pprint.clj");
    pub const CLOJURE_ZIP: &str = include_str!("clj/clojure/zip.clj");
    pub const CLOJURE_DATA_JSON: &str = include_str!("clj/clojure/data/json.clj");
    pub const CLOJURE_TEST: &str = include_str!("clj/clojure/test.clj");
}

/// Read a `deps.edn` and produce the load-path directories it implies, the way
/// the `clojure` CLI builds a classpath. Supported today: `:paths` (relative
/// to the deps.edn's directory) and `:deps {lib {:local/root "…"}}` (the dep's
/// own deps.edn `:paths` when it has one, else `src/`, else the root itself).
/// `:mvn/version` / `:git` coordinates are an ERROR for now — jar/git support
/// is a real feature, not something to silently skip.
pub fn deps_edn_paths<M: ValueModel>(
    rt: &mut Runtime<M>,
    src: &str,
    base: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, String> {
    let form = reader::read_all(rt, src)
        .first()
        .copied()
        .ok_or_else(|| "deps.edn: empty file".to_string())?;
    let top = data::map_entries(rt, form).ok_or_else(|| "deps.edn: not a map".to_string())?;
    let mut out = Vec::new();
    let kw_is = |rt: &Runtime<M>, v: u64, name: &str| -> bool {
        record_field0(rt, v, reader::KEYWORD)
            .is_some_and(|f0| matches!(rt.decode(f0), Val::Sym(s) if rt.sym_name(s) == name))
    };
    let as_string = |rt: &Runtime<M>, v: u64| -> Option<String> { rt.str_view(v).map(|s| s.to_string()) };
    for kv in top.chunks(2) {
        if kv.len() < 2 {
            break;
        }
        if kw_is(rt, kv[0], "paths") {
            let items = data::vector_items(rt, kv[1]).unwrap_or_else(|| rt.list_to_vec(kv[1]));
            for p in items {
                if let Some(s) = as_string(rt, p) {
                    out.push(base.join(s));
                }
            }
        } else if kw_is(rt, kv[0], "deps") {
            let deps = data::map_entries(rt, kv[1])
                .ok_or_else(|| "deps.edn: :deps is not a map".to_string())?;
            for dep in deps.chunks(2) {
                if dep.len() < 2 {
                    continue;
                }
                let coord = data::map_entries(rt, dep[1])
                    .ok_or_else(|| "deps.edn: a dep coordinate is not a map".to_string())?;
                let mut handled = false;
                for ckv in coord.chunks(2) {
                    if ckv.len() < 2 {
                        break;
                    }
                    if kw_is(rt, ckv[0], "local/root") {
                        let root = as_string(rt, ckv[1])
                            .ok_or_else(|| "deps.edn: :local/root must be a string".to_string())?;
                        let root = base.join(root);
                        // the dep's own deps.edn :paths, else src/, else the root
                        let dep_edn = root.join("deps.edn");
                        if dep_edn.is_file() {
                            let dsrc = std::fs::read_to_string(&dep_edn)
                                .map_err(|e| format!("deps.edn: reading {}: {e}", dep_edn.display()))?;
                            out.extend(deps_edn_paths(rt, &dsrc, &root)?);
                        } else if root.join("src").is_dir() {
                            out.push(root.join("src"));
                        } else {
                            out.push(root.clone());
                        }
                        handled = true;
                    } else if kw_is(rt, ckv[0], "mvn/version") {
                        // `lib/name {:mvn/version "…"}` — resolve the jar (plus
                        // transitive compile deps) from ~/.m2 / Maven Central;
                        // jars join the load path directly (ensure_loaded reads
                        // namespaces out of them).
                        let version = as_string(rt, ckv[1])
                            .ok_or_else(|| "deps.edn: :mvn/version must be a string".to_string())?;
                        let libname = match rt.decode(dep[0]) {
                            Val::Sym(s) => rt.sym_name(s).to_string(),
                            _ => return Err("deps.edn: a dep key must be a symbol".to_string()),
                        };
                        let (group, artifact) = match libname.split_once('/') {
                            Some((g, a)) => (g.to_string(), a.to_string()),
                            None => (libname.clone(), libname.clone()),
                        };
                        out.extend(deps::resolve_mvn(&group, &artifact, &version)?);
                        handled = true;
                    } else if kw_is(rt, ckv[0], "git/url") || kw_is(rt, ckv[0], "git/sha") {
                        return Err(
                            ":git deps are not supported yet — use :mvn/version or :local/root"
                                .to_string(),
                        );
                    }
                }
                let _ = handled;
            }
        }
    }
    Ok(out)
}
pub use reader::read_all;

use compile::Compiler;

/// Run a mini-Clojure program (the `clojure.core` prelude first). Returns the
/// last form's value. `require` searches the default load path (`$MICROLANG_PATH`,
/// colon-separated, else `.`).
pub fn run<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    run_with_paths(rt, cs, src, default_load_paths())
}

/// The load-path directories `require` searches (from `$MICROLANG_PATH`, else `.`).
pub fn default_load_paths() -> Vec<std::path::PathBuf> {
    match std::env::var("MICROLANG_PATH") {
        Ok(v) if !v.is_empty() => v.split(':').map(std::path::PathBuf::from).collect(),
        _ => vec![std::path::PathBuf::from(".")],
    }
}

/// Like [`run`], but with an explicit set of load-path directories for `require`.
pub fn run_with_paths<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    src: &str,
    load_paths: Vec<std::path::PathBuf>,
) -> u64 {
    let mut session = Session::new(rt, cs, load_paths);
    session.eval(rt, cs, src)
}

/// A LIVE evaluation session: the loaded prelude plus the compiler/macro state
/// that persists across evals — what a REPL (and an nREPL session) sits on.
/// Create once (loads the prelude), then `eval` any number of times; defs,
/// namespaces, requires, and macros accumulate. Always pass the SAME
/// `Runtime`/backend the session was created with.
pub struct Session {
    macros: HashSet<Sym>,
    comp: Compiler,
}

impl Session {
    pub fn new<M: ValueModel>(
        rt: &mut Runtime<M>,
        cs: &dyn CodeSpace<M>,
        load_paths: Vec<std::path::PathBuf>,
    ) -> Session {
        let mut macros: HashSet<Sym> = HashSet::new();
        let mut comp = Compiler::new(rt);
        comp.set_load_paths(load_paths);
        run_src(rt, cs, &mut macros, &mut comp, sources::CORE);
        // Persistent data structures ported from ClojureScript (EPL-1.0), loaded
        // after the core protocols/shim they build on. Redefines vector/vec/vector?.
        run_src(rt, cs, &mut macros, &mut comp, sources::CLJS_TYPES);
        // The JVM layer — every host class/method/static, as in-language data
        // (`defclass` + `-jvm-registry`). The expander's interop lowering targets
        // these fns; nothing in Rust knows a class name.
        run_src(rt, cs, &mut macros, &mut comp, sources::HOST_JVM);
        // java.io byte streams + clojure.java.io, over the JVM layer.
        run_src(rt, cs, &mut macros, &mut comp, sources::HOST_IO);
        // clojure.core + the cljs types loaded into `clojure.core`; user code from
        // here on runs in the `user` namespace. EVERY var is now ns-qualified, so
        // the frontend's own references to core helpers use `clojure.core/…` names.
        comp.end_core_load();
        // `clojure.string` — bundled, but written ENTIRELY in the language (its
        // `(ns clojure.string)` form sets the ns + marks it loaded). Proof that the
        // string library is library code over one primitive, not builtins.
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_STRING);
        // `clojure.data.json` — a real library, also written entirely in the
        // language (loaded after clojure.string, which its writer uses for `join`).
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_SET);
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_WALK);
        // `clojure.pprint` — print-table only; real libraries (meander) require
        // the namespace for it. See the file for what is deliberately absent.
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_PPRINT);
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_ZIP);
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_DATA_JSON);
        run_src(rt, cs, &mut macros, &mut comp, sources::CLOJURE_TEST);
        comp.set_ns("user");
        // These are provided in-process; `require` must never look for them on disk.
        comp.mark_loaded("clojure.core");
        comp.mark_loaded("clojure.test");
        comp.mark_loaded("user");
        // Route `(obj arg)` for a non-closure record (keyword/map/vector) through
        // the core `-apply-obj` dispatcher, so keywords/collections are callable.
        let apply_obj = rt.intern("clojure.core/-apply-obj");
        rt.set_apply_fn(apply_obj);
        let seq_sym = rt.intern("clojure.core/seq");
        rt.set_seq_fn(seq_sym);
        Session { macros, comp }
    }

    /// Evaluate source in this session, returning the last form's (realized)
    /// value. Compiler/macro state persists to the next call.
    pub fn eval<M: ValueModel>(
        &mut self,
        rt: &mut Runtime<M>,
        cs: &dyn CodeSpace<M>,
        src: &str,
    ) -> u64 {
        // Install the reader+compiler re-entry bridge (read-string/eval/
        // macroexpand-1) with raw pointers to the live state. Valid only for this
        // call; cleared before returning. See `FrontendBridge`.
        // SAFETY: launder cs's lifetime to a raw pointer; the bridge is cleared
        // before this call returns, so the pointer never outlives `cs`.
        let cs_ptr: *const dyn CodeSpace<M> =
            unsafe { std::mem::transmute::<&dyn CodeSpace<M>, &'static dyn CodeSpace<M>>(cs) };
        let bridge = FrontendBridge::<M> {
            comp: &mut self.comp as *mut Compiler,
            macros: &mut self.macros as *mut HashSet<Sym>,
            cs: cs_ptr,
        };
        rt.set_eval_bridge(Arc::new(bridge));
        let result = run_src(rt, cs, &mut self.macros, &mut self.comp, src);
        rt.clear_eval_bridge();
        // Force any lazy sequence in the final value so callers / the printer
        // (which can't invoke thunks) see a fully realized result.
        let slot = rt.push_root(result);
        let realize = rt.intern("clojure.core/-realize");
        let out = match rt.global(realize) {
            Some(rf) => cs.invoke(cs, rt, rf, &[rt.root_get(slot)]),
            None => rt.root_get(slot),
        };
        rt.truncate_roots(slot);
        out
    }

    /// The namespace subsequent evals compile in (the REPL prompt).
    pub fn current_ns(&self) -> &str {
        self.comp.current_ns()
    }
}

fn run_src<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    src: &str,
) -> u64 {
    let forms = reader::read_all(rt, src);
    let base = rt.root_depth();
    for &f in &forms {
        rt.push_root(f);
    }
    let mut last = rt.encode(Val::Nil);
    for i in 0..forms.len() {
        let f = rt.root_get(base + i);
        last = eval_form(rt, cs, macros, comp, f, true);
    }
    rt.truncate_roots(base);
    last
}

/// Expand, compile straight to `Ir`, and run it. The toolkit's `analyze` is
/// never used — this frontend owns the surface -> `Ir` lowering.
///
/// `boundary`: at the true top level a still-pending signal is an UNCAUGHT
/// throw — a program error that terminates (like an uncaught exception in
/// real Clojure). Inside the eval BRIDGE the signal must PROPAGATE instead,
/// so `(try (eval …) (catch …))` catches it.
fn eval1<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
    boundary: bool,
) -> u64 {
    let expanded = expand(rt, cs, macros, comp, form);
    let slot = rt.push_root(expanded);
    let ir = comp.compile(rt, rt.root_get(slot));
    rt.truncate_roots(slot);
    // Closure-convert to the flat shape the tiers execute.
    let ir = microlang::flatten(&ir);
    let r = cs.eval_ir(cs, rt, &ir, &None);
    if boundary && rt.pending() {
        let sig = rt.take_signal();
        if sig.kind == 1 {
            panic!("uncaught throw: {}", rt.print(sig.value));
        }
        panic!("escape continuation invoked outside its (%callec) extent");
    }
    r
}

/// One macro-expansion step: if `form` is `(macro …)`, invoke the macro once and
/// return the result; otherwise return `form` unchanged. No recursion, no
/// structural desugaring (that's `expand`). Backs `macroexpand-1`.
fn macroexpand_1_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let Some((head, _)) = rt.as_cons(form) else { return form };
    let Val::Sym(hs) = rt.decode(head) else { return form };
    let q = comp.resolve_ref(rt, hs);
    if !macros.contains(&q) {
        return form;
    }
    let mfn = match rt.global(q) {
        Some(v) => v,
        None => return form,
    };
    let args = rt.list_to_vec(form);
    let nilv = rt.encode(Val::Nil);
    let mut margs = vec![form, nilv];
    margs.extend_from_slice(&args[1..]);
    cs.invoke(cs, rt, mfn, &margs)
}

/// The reader+compiler re-entry bridge for `read-string`/`eval`/`macroexpand-1`
/// (see `microlang::EvalBridge`). Holds RAW pointers to the live compiler state
/// owned by `run_with_paths`; valid only while that call is on the stack (the
/// bridge is cleared before it returns). Dereferenced only on the installing (main)
/// thread, inside `eval_ir`, where the outer frame is not touching the compiler —
/// so the aliasing is benign in practice.
struct FrontendBridge<M: ValueModel> {
    comp: *mut Compiler,
    macros: *mut HashSet<Sym>,
    cs: *const dyn CodeSpace<M>,
}
unsafe impl<M: ValueModel> Send for FrontendBridge<M> {}
unsafe impl<M: ValueModel> Sync for FrontendBridge<M> {}
impl<M: ValueModel> EvalBridge<M> for FrontendBridge<M> {
    fn read_string(&self, rt: &mut Runtime<M>, s: u64) -> u64 {
        let src = rt.as_str(s, "read-string");
        let form = reader::read_all(rt, &src)
            .first()
            .copied()
            .unwrap_or_else(|| rt.encode(Val::Nil));
        // `::kw` resolves at read time in Clojure — do it here so the returned
        // datum contains real keywords, wherever it flows next.
        let comp: &Compiler = unsafe { &*self.comp };
        resolve_auto_keywords(rt, comp, form)
    }
    fn eval(&self, rt: &mut Runtime<M>, form: u64) -> u64 {
        // Route through eval_form so eval'd code gets the full TOP-LEVEL
        // treatment — `ns`/`require` switch the compiler, `defmacro` registers,
        // `(do …)` sequences — exactly like source forms. Non-boundary: a
        // pending signal (uncaught throw) PROPAGATES, so `(try (eval …))` works.
        let cs: &dyn CodeSpace<M> = unsafe { &*self.cs };
        let macros: &mut HashSet<Sym> = unsafe { &mut *self.macros };
        let comp: &mut Compiler = unsafe { &mut *self.comp };
        eval_form(rt, cs, macros, comp, form, false)
    }
    fn macroexpand_1(&self, rt: &mut Runtime<M>, form: u64) -> u64 {
        let cs: &dyn CodeSpace<M> = unsafe { &*self.cs };
        let macros: &HashSet<Sym> = unsafe { &*self.macros };
        let comp: &Compiler = unsafe { &*self.comp };
        macroexpand_1_form(rt, cs, macros, comp, form)
    }
    fn current_ns(&self, rt: &mut Runtime<M>) -> u64 {
        let comp: &Compiler = unsafe { &*self.comp };
        let s = rt.intern(comp.current_ns());
        rt.encode(Val::Sym(s))
    }
}

/// Resolve `::foo` / `::alias/foo` markers (reader `KeywordAutoNs` records) to
/// plain namespaced keywords per the CURRENT namespace — evaluation is
/// form-by-form, so any preceding `ns`/`alias` forms have already run, giving
/// Clojure's read-time resolution semantics. Rebuilds only when a marker is
/// present; quoted data included (resolution happens before quote is literal).
fn resolve_auto_keywords<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    if !has_auto_keyword(rt, form) {
        return form;
    }
    rebuild_auto_keywords(rt, comp, form)
}

fn has_auto_keyword<M: ValueModel>(rt: &Runtime<M>, form: u64) -> bool {
    if let Val::Ref(id) = rt.decode(form) {
        match rt.view_gc(id) {
            ObjView::Record { type_id, fields } => {
                if rt.sym_name(type_id) == reader::KEYWORD_AUTO_NS {
                    return true;
                }
                fields.iter().any(|&f| has_auto_keyword(rt, f))
            }
            ObjView::Cons { head, tail } => has_auto_keyword(rt, head) || has_auto_keyword(rt, tail),
            ObjView::Vector { elems, .. } => elems.iter().any(|&e| has_auto_keyword(rt, e)),
            _ => false,
        }
    } else {
        false
    }
}

fn rebuild_auto_keywords<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let Val::Ref(id) = rt.decode(form) else { return form };
    // Copy the shape out of the view first: the recursive calls below need
    // `&mut rt`, which a live borrow from `view_gc` would block.
    enum Shape {
        AutoNsKeyword(Vec<u64>),
        Record(Sym, Vec<u64>),
        Cons(u64, u64),
        Vector(Vec<u64>),
        Other,
    }
    let shape = match rt.view_gc(id) {
        ObjView::Record { type_id, fields } if rt.sym_name(type_id) == reader::KEYWORD_AUTO_NS => {
            Shape::AutoNsKeyword(fields.to_vec())
        }
        ObjView::Record { type_id, fields } => Shape::Record(type_id, fields.to_vec()),
        ObjView::Cons { head, tail } => Shape::Cons(head, tail),
        ObjView::Vector { elems, .. } => Shape::Vector(elems.to_vec()),
        _ => Shape::Other,
    };
    match shape {
        Shape::AutoNsKeyword(fields) => {
            let name = match rt.decode(fields[0]) {
                Val::Sym(s) => rt.sym_name(s).to_string(),
                _ => panic!("::keyword marker: name must be a symbol"),
            };
            // `::foo` -> current ns; `::alias/foo` -> the alias's namespace.
            let full = match name.split_once('/') {
                Some((alias, n)) => {
                    let ns = comp.alias_target(alias).unwrap_or_else(|| {
                        panic!("Invalid token: ::{name} (no such namespace alias: {alias})")
                    });
                    format!("{ns}/{n}")
                }
                None => format!("{}/{name}", comp.current_ns()),
            };
            // Interned like any other keyword: `::foo` and the `:cur.ns/foo`
            // it resolves to MUST be the same object.
            let s = rt.intern(&full);
            rt.intern_keyword(s)
        }
        Shape::Record(type_id, fields) => {
            let nf: Vec<u64> = fields.iter().map(|&f| rebuild_auto_keywords(rt, comp, f)).collect();
            let rid = rt.alloc_record(type_id, &nf);
            <M::R as microlang::Repr>::enc_ref(rid)
        }
        Shape::Cons(head, tail) => {
            let h = rebuild_auto_keywords(rt, comp, head);
            let t = rebuild_auto_keywords(rt, comp, tail);
            let rid = rt.alloc(Obj::Cons { head: h, tail: t });
            <M::R as microlang::Repr>::enc_ref(rid)
        }
        Shape::Vector(elems) => {
            let ne: Vec<u64> = elems.iter().map(|&e| rebuild_auto_keywords(rt, comp, e)).collect();
            let rid = rt.alloc_vector(&ne);
            <M::R as microlang::Repr>::enc_ref(rid)
        }
        Shape::Other => form,
    }
}

fn eval_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
    boundary: bool,
) -> u64 {
    // `::kw` markers resolve against the CURRENT namespace, before anything
    // else dissects the form (Clojure resolves them at read time).
    let form = resolve_auto_keywords(rt, comp, form);
    // A TOP-LEVEL `(do …)` is a sequence of top-level forms (exactly Clojure's
    // compiler): `ns`/`defmacro`/`definline` inside it get their eval-level
    // handling. This is what lets a REPL message body switch namespaces.
    if let Some((h, _)) = rt.as_cons(form) {
        if is_sym(rt, h, "do") {
            let items = rt.list_to_vec(form);
            let base = rt.root_depth();
            for &f in &items[1..] {
                rt.push_root(f);
            }
            let mut last = rt.encode(Val::Nil);
            for i in 0..items.len() - 1 {
                let f = rt.root_get(base + i);
                last = eval_form(rt, cs, macros, comp, f, boundary);
                // a pending (propagating) signal aborts the sequence
                if rt.pending() {
                    break;
                }
            }
            rt.truncate_roots(base);
            return last;
        }
    }
    // Namespace declarations mutate the compiler's resolution state (and may LOAD
    // required files) and yield nil. Handled before macro/def checks.
    if let Some(r) = handle_ns_form(rt, cs, macros, comp, form) {
        return r;
    }
    // Real core.clj: `(def ^{:macro true} name (fn ...))` — reader wrapped the
    // name as `(-macro-meta name)`. Define the fn, then register the macro under
    // its RESOLVED (namespace-qualified) sym.
    if let Some((name, newform)) = strip_def_macro_meta(rt, form) {
        let r = eval1(rt, cs, macros, comp, newform, boundary);
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
        return r;
    }
    // Real core.clj registers a macro AFTER defining it: `(. (var foo) (setMacro))`.
    if let Some(name) = setmacro_target(rt, form) {
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
        return rt.encode(Val::Nil);
    }
    // `(definline name [args] `template)` — an inline fn IS a macro in this
    // dialect (the template splices at call sites); registration must happen
    // here at eval level, exactly like defmacro. Call-position only (real
    // definline also defs a fn; none of the code we run passes them around).
    if let Some((h, _)) = rt.as_cons(form) {
        if is_sym(rt, h, "definline") {
            let items = rt.list_to_vec(form);
            // the name may arrive wrapped by reader meta (`^:private read-byte`)
            let mut name = items[1];
            while let Some((mh, _)) = rt.as_cons(name) {
                let parts = rt.list_to_vec(name);
                if parts.len() == 2
                    && (is_sym(rt, mh, "-private-meta")
                        || is_sym(rt, mh, "-macro-meta")
                        || is_sym(rt, mh, "-dynamic-meta"))
                {
                    name = parts[1];
                } else {
                    break;
                }
            }
            let dm = sym(rt, "defmacro");
            let mut out = vec![dm, name];
            // skip an optional docstring before the param vector
            let mut rest = &items[2..];
            if rest.len() > 1 && rt.str_view(rest[0]).is_some() {
                rest = &rest[1..];
            }
            out.extend_from_slice(rest);
            let rewritten = rt.vec_to_list(&out);
            return eval_form(rt, cs, macros, comp, rewritten, boundary);
        }
    }
    if let Some(name) = defmacro_name(rt, form) {
        // (defmacro name params body...) -> (def name (fn [&form &env params...] body...)).
        // Every macro fn gets the Clojure `&form`/`&env` hidden params (our macros
        // just ignore them), so our expander invokes all macros uniformly.
        let mut items = rt.list_to_vec(form);
        // Skip an optional docstring (a String) then attr-map (a Map) between the
        // name and the param list — `(defmacro m "doc" {…} [params] …)`.
        let is_str = |rt: &Runtime<M>, x: u64| rt.str_view(x).is_some();
        if items.len() > 3 && is_str(rt, items[2]) {
            items.remove(2);
        }
        if items.len() > 3 && data::is_map_rep(rt, items[2]) {
            items.remove(2);
        }
        let form_s = sym(rt, "&form");
        let env_s = sym(rt, "&env");
        let fn_sym = sym(rt, "fn");
        // Prepend the hidden `&form`/`&env` params. Params come as a vector `[a b]`
        // OR our legacy list style `(a b & c)`.
        let with_hidden = |rt: &mut Runtime<M>, pv: u64| -> u64 {
            let orig = data::vector_items(rt, pv).unwrap_or_else(|| rt.list_to_vec(pv));
            let mut params = vec![form_s, env_s];
            params.extend(orig);
            make_vector(rt, params)
        };
        // Multi-arity ONLY when `items[2]` is a clause list `([params] …)` — i.e. a
        // list whose FIRST element is a param vector (a bare-symbol list is legacy
        // single-arity params).
        let multi = data::vector_items(rt, items[2]).is_none()
            && rt.as_cons(items[2]).is_some()
            && rt
                .list_to_vec(items[2])
                .first()
                .is_some_and(|&f| data::vector_items(rt, f).is_some());
        let lam = if !multi {
            // single-arity: (defmacro name [params] body…) or (name (a b) body…)
            let pv = with_hidden(rt, items[2]);
            let mut fnform = vec![fn_sym, pv];
            fnform.extend_from_slice(&items[3..]);
            rt.vec_to_list(&fnform)
        } else {
            // multi-arity: (defmacro name ([params] body…) …) — one clause per arity.
            let mut fnform = vec![fn_sym];
            for &clause in &items[2..] {
                let cparts = rt.list_to_vec(clause);
                let pv = with_hidden(rt, cparts[0]);
                let mut newclause = vec![pv];
                newclause.extend_from_slice(&cparts[1..]);
                fnform.push(rt.vec_to_list(&newclause));
            }
            rt.vec_to_list(&fnform)
        };
        let def_sym = sym(rt, "def");
        let defform = rt.vec_to_list(&[def_sym, items[1], lam]);
        let r = eval1(rt, cs, macros, comp, defform, boundary);
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
        return r;
    }
    eval1(rt, cs, macros, comp, form, boundary)
}

/// Fully macro-expand + desugar a form.
fn expand<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let slot = RootVec::one(rt, form);
    // 1. head-expand while the head resolves to a macro
    loop {
        // A form built by a macro / lazy seq op may be (or contain in its tail)
        // a LazySeq — realize the spine so the cons walks below see all of it.
        let f = force_spine(rt, cs, slot.get(rt, 0));
        slot.set(rt, 0, f);
        let Some((head, _)) = rt.as_cons(f) else { break };
        let Val::Sym(hs) = rt.decode(head) else { break };
        // Resolve the head to its namespace-qualified var; is that a macro? This
        // makes bare, fully-qualified, AND alias-qualified macro calls all work.
        let q = comp.resolve_ref(rt, hs);
        if !macros.contains(&q) {
            break;
        }
        let mfn = match rt.global(q) {
            Some(v) => v,
            None => break,
        };
        // Clojure macro convention: (&form &env & args). Pass the whole form and
        // a nil env before the argument forms.
        let args = rt.list_to_vec(f);
        let nilv = rt.encode(Val::Nil);
        let mut margs = vec![f, nilv];
        margs.extend_from_slice(&args[1..]);
        let result = cs.invoke(cs, rt, mfn, &margs);
        slot.set(rt, 0, result);
    }
    // 2. structural desugar / recurse
    let f = slot.get(rt, 0);
    let out = if let Some((head, _)) = rt.as_cons(f) {
        if is_sym(rt, head, "quote") {
            // Quoted data is LITERAL, exactly as in Clojure: collection literals
            // already ARE the runtime persistent collections (the reader builds
            // them), so the whole datum compiles to a `Const` untouched.
            f
        } else if is_sym(rt, head, "syntax-quote") {
            // ` template -> a form that BUILDS the data; then expand that.
            let inner = rt.list_to_vec(f)[1];
            let mut gs = HashMap::new();
            let built = syntax_quote(rt, comp, inner, &mut gs);
            expand(rt, cs, macros, comp, built)
        } else if is_sym(rt, head, "ns") || is_sym(rt, head, "in-ns") {
            rt.encode(Val::Nil) // namespaces are a no-op for now
        } else if is_sym(rt, head, "fn") || is_sym(rt, head, "fn*") {
            expand_fn(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "let") || is_sym(rt, head, "let*") {
            rebuild_binder(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "loop") || is_sym(rt, head, "loop*") {
            expand_loop(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "defprotocol") || is_sym(rt, head, "definterface") {
            // `definterface` is treated like a protocol (its marker interfaces just
            // register the name; any methods become protocol methods).
            let d = desugar_defprotocol(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "extend-type") {
            let d = desugar_extend_type(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "deftype") {
            let d = desugar_deftype(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "var") {
            // `(var x)` / `#'x` -> a first-class Var handle `(record 'Var 'ns/x)`.
            // The name is RESOLVED so the handle keys the global table exactly as
            // an ordinary reference to `x` does — deref/alter-var-root operate by
            // that sym. (The `(. (var x) (setMacro))` bootstrap is intercepted
            // earlier, at eval-form level; this handles `var` elsewhere.)
            let items = rt.list_to_vec(f);
            let resolved = match rt.decode(items[1]) {
                // var-quote may reach PRIVATE vars (Clojure allows `#'ns/private`).
                Val::Sym(s) => rt.encode(Val::Sym(comp.resolve_ref_allow_private(rt, s))),
                _ => items[1],
            };
            let rec = sym(rt, "record");
            let vtag = sym(rt, "Var");
            let tag_q = quote_form(rt, vtag);
            let name_q = quote_form(rt, resolved);
            let g = rt.vec_to_list(&[rec, tag_q, name_q]);
            expand(rt, cs, macros, comp, g)
        } else if let Some(qualified) = resolve_rewrite(rt, comp, f) {
            // `(resolve 'x)` / `(ns-resolve 'ns 'x)` with LITERAL symbols: namespace
            // resolution is compile-time, so rewrite to `(find-var 'qualified)`.
            let find = sym(rt, "find-var");
            let qsym = rt.encode(Val::Sym(qualified));
            let qform = quote_form(rt, qsym);
            let g = rt.vec_to_list(&[find, qform]);
            expand(rt, cs, macros, comp, g)
        } else if is_sym(rt, head, "new") {
            // `(new Class args…)` is the same as `(Class. args…)` — rewrite to the
            // trailing-dot constructor form and let the interop shim handle it.
            let items = rt.list_to_vec(f);
            let cname = sym_str(rt, items[1]);
            let ctor = sym(rt, &format!("{cname}."));
            let mut out = vec![ctor];
            out.extend_from_slice(&items[2..]);
            let d = rt.vec_to_list(&out);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "instance?") {
            let d = instance_rewrite(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "try") {
            // Desugar typed multi-catch into a single catch-all whose body is a
            // type-dispatch (ClojureScript's model); expand the result.
            let d = desugar_try(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "binding") {
            // `(binding [*x* v] body…)` — thread-local dynamic-var bindings via a
            // `try`/`finally` that installs and (always) unwinds them.
            let d = desugar_binding(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "with-redefs") {
            // `(with-redefs [f g] body…)` — temporarily rebind var ROOTS (for
            // testing), restoring them in a `finally`.
            let d = desugar_with_redefs(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if record_field0(rt, head, reader::KEYWORD).is_some() {
            // keyword in head position: (:k m) -> (get m :k)
            let items = rt.list_to_vec(f);
            let getsym = sym(rt, "get");
            let g = rt.vec_to_list(&[getsym, items[1], head]);
            expand(rt, cs, macros, comp, g)
        } else if let Some(rw) = interop_rewrite(rt, comp, f) {
            expand(rt, cs, macros, comp, rw)
        } else {
            // Ordinary call `(f args…)`: every element has to survive the
            // expansion of every OTHER element (each one can invoke a macro).
            let its = rt.list_to_vec(f);
            let mut items = RootVec::new(rt, &its);
            expand_each(rt, cs, macros, comp, &mut items);
            let snap = items.snapshot(rt);
            let out = rt.vec_to_list(&snap);
            items.release(rt);
            out
        }
    } else if constant_literal(rt, f) {
        // A collection literal whose elements are ALL constants IS a constant:
        // quote it, so the datum the reader already built is the value. This is
        // Clojure's own rule (`Compiler.java`: a vector/map/set expression with
        // constant parts parses to a `ConstantExpr`), and it matters — `[]` in
        // `(get m k [])` was rebuilding an empty vector through a real `vector`
        // call PER ELEMENT (~390ns/op, the top cost in group-by).
        //
        // Sharing one object across evaluations is sound because these
        // collections are immutable: conj/assoc path-copy, `-with-meta` builds a
        // new record, `caching-hash` recomputes (no ^:mutable field), and
        // `transient` copies the tail + treats un-stamped nodes as copy-on-write.
        // Identity now matches Clojure's too (`(identical? (f) (f))` for
        // `(fn [] [])` is true on the JVM).
        quote_form(rt, f)
    } else if let Some(elems) = binding_items(rt, f) {
        // A vector in EXPRESSION position -> `(vector e0 e1 …)` so its elements
        // evaluate. `binding_items` matches every vector representation (the
        // reader's phase type and anything a macro/syntax-quote returned).
        // Param/binding vectors are handled by fn/let/loop BEFORE reaching here,
        // so they are safe.
        let mut elems = RootVec::new(rt, &elems);
        let r = build_call(rt, cs, macros, comp, "vector", &mut elems);
        elems.release(rt);
        r
    } else if let Some(kvs) = data::map_entries(rt, f) {
        let mut kvs = RootVec::new(rt, &kvs);
        let r = build_call(rt, cs, macros, comp, "hash-map", &mut kvs);
        kvs.release(rt);
        r
    } else if let Some(es) = data::set_items(rt, f) {
        let mut es = RootVec::new(rt, &es);
        let r = build_call(rt, cs, macros, comp, "hash-set", &mut es);
        es.release(rt);
        r
    } else {
        f // keyword / string / char / number / symbol — self-evaluating
    };
    slot.release(rt);
    out
}

/// Is `f` a COLLECTION literal all of whose elements are themselves constants —
/// i.e. would evaluating it just rebuild the datum the reader already made?
///
/// Mirrors `expand`'s own classification, in its order: a cons list is a call or
/// special form and a symbol is a variable reference (neither is constant);
/// a vector/map/set is constant exactly when every element is; anything else
/// (number, string, char, keyword, bool, nil, `()`) is self-evaluating and so
/// already constant. Only COLLECTIONS are asked — the scalars reach `expand`'s
/// self-evaluating arm regardless, and answering true for them here would wrap
/// them in a pointless `quote`.
///
/// REPRESENTATION GATE (`data::is_final_rep`) — the reason this is not simply
/// "is it a collection of constants". Quoting a datum is only equivalent to
/// evaluating its constructor call when the datum IS what that call would build.
/// The reader builds the representation of the phase it runs IN, but the
/// constructor runs LATER: `clojure.core` is read while only core's own `PVec`
/// exists, yet its bodies run once `cljs_types` has installed
/// `PersistentVector`. Freezing those datums hands a `PVec` back out of
/// `group-by` where every other vector in the system is a `PersistentVector`
/// (observed — this is not hypothetical). The ambient phase does NOT decide it
/// either: core datums baked into MACRO TEMPLATES are expanded in user phase and
/// are still phase-1 objects. So every collection, at every nesting depth, must
/// itself already be in the final representation; a bootstrap-phase literal
/// keeps the constructor-call route and is unchanged.
fn constant_literal<M: ValueModel>(rt: &Runtime<M>, f: u64) -> bool {
    fn is_const<M: ValueModel>(rt: &Runtime<M>, f: u64) -> bool {
        if rt.as_cons(f).is_some() {
            return false; // a call / special form
        }
        if matches!(rt.decode(f), Val::Sym(_)) {
            return false; // a variable reference
        }
        let is_coll = binding_items(rt, f).is_some()
            || data::map_entries(rt, f).is_some()
            || data::set_items(rt, f).is_some();
        if !is_coll {
            return true; // number / string / char / keyword / bool / nil / `()`
        }
        if !data::is_final_rep(rt, f) {
            return false; // a bootstrap-phase datum: its constructor rebuilds it
        }
        if let Some(elems) = binding_items(rt, f) {
            return elems.iter().all(|&e| is_const(rt, e));
        }
        if let Some(kvs) = data::map_entries(rt, f) {
            return kvs.iter().all(|&e| is_const(rt, e));
        }
        let es = data::set_items(rt, f).expect("collection with no reader");
        es.iter().all(|&e| is_const(rt, e))
    }
    let is_coll = binding_items(rt, f).is_some()
        || data::map_entries(rt, f).is_some()
        || data::set_items(rt, f).is_some();
    is_coll && is_const(rt, f)
}

/// If `form` is a lazy seq — or a cons list whose TAIL hits a lazy node — force
/// its spine into an eager cons list (via core's `-force-spine`), so the
/// expander/compiler can walk it. Elements are left untouched (they are forced
/// on recursion when they sit in code position). Anything else passes through.
/// Before `-force-spine` exists (early core bootstrap) no macro returns lazy
/// seqs, so the form passes through.
fn force_spine<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, form: u64) -> u64 {
    let mut f = form;
    let lazy = loop {
        match rt.decode(f) {
            Val::Ref(id) => match rt.view_gc(id) {
                ObjView::Cons { tail, .. } => f = tail,
                ObjView::Record { type_id, .. } if rt.sym_name(type_id) == "LazySeq" => break true,
                _ => break false,
            },
            _ => break false,
        }
    };
    if !lazy {
        return form;
    }
    let fs = rt.intern("clojure.core/-force-spine");
    match rt.global(fs) {
        Some(g) => cs.invoke(cs, rt, g, &[form]),
        None => form,
    }
}

/// Expand every form in `items`, IN PLACE in its shadow slot.
///
/// THE rooting choke point of the expander. Expanding form `i` invokes macros —
/// a safepoint, which relocates every OTHER form in the list: both the ones not
/// yet expanded and the results already produced. So the whole list lives in the
/// shadow stack for the duration of the walk and each slot is re-read at the
/// point of use. Taking `&mut RootVec` (not `&[u64]`) is what makes this
/// impossible to call wrong: a caller with a bare `Vec<u64>` must root it first.
fn expand_each<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    items: &mut RootVec,
) {
    for i in 0..items.len() {
        let it = items.get(rt, i);
        let ex = expand(rt, cs, macros, comp, it);
        items.set(rt, i, ex);
    }
}

/// `(vector|hash-map|hash-set <elems>)` from the element forms.
fn build_call<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    ctor: &str,
    args: &mut RootVec,
) -> u64 {
    expand_each(rt, cs, macros, comp, args);
    // Interned so AFTER the expansion: `fsym` is an immediate and could not move
    // anyway, but building it here keeps the snapshot the only cross-cutting read.
    let fsym = sym(rt, ctor);
    let mut out = vec![fsym];
    out.extend(args.snapshot(rt));
    rt.vec_to_list(&out)
}

/// `(let/loop [binds] body)` or `(fn [params] body)`: normalize the binding
/// VECTOR to the toolkit's binding LIST, desugaring DESTRUCTURING, then expand.
fn rebuild_binder<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let its = rt.list_to_vec(form);
    // The form's own items must survive the `expand_each` calls below (macro
    // invocation = a safepoint = a move). `its[2..]` — the body — is read AFTER
    // one of them on the let/loop path, so the whole list is rooted up front.
    let items = RootVec::new(rt, &its);
    let bind_forms = binding_items(rt, its[1]).unwrap_or_else(|| rt.list_to_vec(its[1]));

    if is_sym(rt, its[0], "fn") || is_sym(rt, its[0], "fn*") {
        // Params: a destructuring param becomes a fresh param + a `let` in the
        // body binding the pattern to it. (Symbols and `&` pass through.)
        let mut params = Vec::new();
        let mut wrap = Vec::new(); // pattern, init pairs
        let mut prev_amp = false;
        for &p in &bind_forms {
            if matches!(rt.decode(p), Val::Sym(_)) {
                params.push(p);
                prev_amp = is_sym(rt, p, "&");
            } else {
                let g = gensym(rt, "p");
                params.push(g);
                // `& {:keys …}` — the trailing args are keyword pairs; collect them
                // into a map before destructuring the pattern.
                let init = if prev_amp && data::is_map_rep(rt, p) {
                    let kw = sym(rt, "-kwargs->map");
                    rt.vec_to_list(&[kw, g])
                } else {
                    g
                };
                wrap.push(p);
                wrap.push(init);
                prev_amp = false;
            }
        }
        let paramlist = rt.vec_to_list(&params);
        let body: Vec<u64> = its[2..].to_vec();
        let inner = if wrap.is_empty() {
            body
        } else {
            // (let [pat0 g0 ..] body...)
            let letk = sym(rt, "let");
            let bvec = make_vector(rt, wrap);
            let mut letf = vec![letk, bvec];
            letf.extend(body);
            vec![rt.vec_to_list(&letf)]
        };
        // `paramlist` is a freshly built cons list — it must survive the body's
        // expansion below, which invokes macros.
        let held = RootVec::one(rt, paramlist);
        let mut inner = RootVec::new(rt, &inner);
        expand_each(rt, cs, macros, comp, &mut inner);
        let mut out = vec![items.get(rt, 0), held.get(rt, 0)];
        out.extend(inner.snapshot(rt));
        let r = rt.vec_to_list(&out);
        inner.release(rt);
        held.release(rt);
        items.release(rt);
        return r;
    }

    // let / loop: destructure each (pattern, init) pair.
    let mut binds = Vec::new();
    let mut i = 0;
    while i + 1 < bind_forms.len() {
        let pairs = destructure(rt, bind_forms[i], bind_forms[i + 1]);
        binds.extend(pairs);
        i += 2;
    }
    let mut binds = RootVec::new(rt, &binds);
    expand_each(rt, cs, macros, comp, &mut binds);
    let snap = binds.snapshot(rt);
    let bindlist = rt.vec_to_list(&snap);
    binds.release(rt);
    // `bindlist` (just built) and the BODY forms (read out of `items`, which the
    // binding expansion above may already have relocated) both have to survive
    // the second expansion.
    let held = RootVec::one(rt, bindlist);
    let body: Vec<u64> = (2..items.len()).map(|i| items.get(rt, i)).collect();
    let mut body = RootVec::new(rt, &body);
    expand_each(rt, cs, macros, comp, &mut body);
    let mut out = vec![items.get(rt, 0), held.get(rt, 0)];
    out.extend(body.snapshot(rt));
    let r = rt.vec_to_list(&out);
    body.release(rt);
    held.release(rt);
    items.release(rt);
    r
}

/// Desugar one binding `(pat, init)` into a flat `[sym expr sym expr ...]` list
/// of simple bindings. Handles symbols, sequential `[a b & r]` (nested), and
/// `{:keys [x y]}` map destructuring.
fn destructure<M: ValueModel>(rt: &mut Runtime<M>, pat: u64, init: u64) -> Vec<u64> {
    if matches!(rt.decode(pat), Val::Sym(_)) {
        return vec![pat, init];
    }
    if let Some(elems) = binding_items(rt, pat) {
        let t = gensym(rt, "vec");
        let mut binds = vec![t, init];
        let mut idx: i128 = 0;
        let mut k = 0;
        while k < elems.len() {
            if is_sym(rt, elems[k], "&") {
                let iv = int(rt, idx);
                let rest_expr = call2(rt, "drop", iv, t);
                // `& {:keys …}` — collect trailing kwargs into a map to destructure.
                let rest_expr = if data::is_map_rep(rt, elems[k + 1]) {
                    let kw = sym(rt, "-kwargs->map");
                    rt.vec_to_list(&[kw, rest_expr])
                } else {
                    rest_expr
                };
                binds.extend(destructure(rt, elems[k + 1], rest_expr));
                k += 2;
                continue;
            }
            if is_keyword(rt, elems[k], "as") {
                // [a b :as whole] -> whole bound to the entire collection.
                binds.push(elems[k + 1]);
                binds.push(t);
                k += 2;
                continue;
            }
            if !is_sym(rt, elems[k], "_") {
                let iv = int(rt, idx);
                // `(nth t idx nil)` — a missing index binds nil (Clojure destructuring
                // semantics), not an out-of-bounds error.
                let nthk = sym(rt, "nth");
                let niln = rt.encode(Val::Nil);
                let nth_expr = rt.vec_to_list(&[nthk, t, iv, niln]);
                binds.extend(destructure(rt, elems[k], nth_expr));
            }
            idx += 1;
            k += 1;
        }
        return binds;
    }
    if let Some(kvs) = data::map_entries(rt, pat) {
        let t = gensym(rt, "map");
        let mut binds = vec![t, init];
        // First locate `:or` defaults (a `{sym default …}` map) so any binding
        // can fall back when its key is absent.
        let mut defaults: Vec<(u64, u64)> = Vec::new();
        let mut k = 0;
        while k + 1 < kvs.len() {
            if is_keyword(rt, kvs[k], "or") {
                if let Some(dkvs) = data::map_entries(rt, kvs[k + 1]) {
                    let mut j = 0;
                    while j + 1 < dkvs.len() {
                        defaults.push((dkvs[j], dkvs[j + 1]));
                        j += 2;
                    }
                }
            }
            k += 2;
        }
        // `get t key`, or `(if (contains? t key) (get t key) default)` if the
        // symbol has an `:or` default.
        let get_with_default =
            |rt: &mut Runtime<M>, name: u64, key: u64| -> u64 {
                let ge = call2(rt, "get", t, key);
                for &(dn, dv) in &defaults {
                    if let (Val::Sym(a), Val::Sym(b)) = (rt.decode(dn), rt.decode(name)) {
                        if a == b {
                            let has = call2(rt, "contains?", t, key);
                            let ifs = sym(rt, "if");
                            return rt.vec_to_list(&[ifs, has, ge, dv]);
                        }
                    }
                }
                ge
            };
        let mut k = 0;
        while k + 1 < kvs.len() {
            let key = kvs[k];
            let val = kvs[k + 1];
            if let Some(dir_ns) = keys_directive(rt, key) {
                // {:keys [x y]}    -> x (get t :x), y (get t :y)   (honoring :or)
                // {:keys [ns/x]}   -> x (get t :ns/x)      — the NAME is the local
                // {:a/keys [x]}    -> x (get t :a/x)       — the directive's ns
                // Both namespaced forms are Clojure 1.9+, and real libraries use
                // them (meander/epsilon does; it is why this rejected its source
                // with "let: binding name must be a symbol" — it tried to bind
                // `ns/x` as a local).
                if let Some(vl) = binding_items(rt, val) {
                    for s in vl {
                        let (local, full) = split_ns_binding(rt, s, dir_ns.as_deref());
                        let kw = keyword_expr(rt, full);
                        let ge = get_with_default(rt, local, kw);
                        binds.push(local);
                        binds.push(ge);
                    }
                }
            } else if is_keyword(rt, key, "strs") {
                // {:strs [a b]} -> a (get t "a"), b (get t "b")  (string keys)
                if let Some(vl) = binding_items(rt, val) {
                    for s in vl {
                        let name = match rt.decode(s) {
                            Val::Sym(sy) => rt.sym_name(sy).to_string(),
                            _ => continue,
                        };
                        let id = rt.alloc(Obj::Str(name));
                        let strv = <M::R as microlang::Repr>::enc_ref(id);
                        let ge = get_with_default(rt, s, strv);
                        binds.push(s);
                        binds.push(ge);
                    }
                }
            } else if let Some(dir_ns) = syms_directive(rt, key) {
                // {:syms [a b]}  -> a (get t 'a)     (symbol keys)
                // {:a/syms [x]}  -> x (get t 'a/x)   — same ns rules as :keys
                if let Some(vl) = binding_items(rt, val) {
                    for s in vl {
                        let (local, full) = split_ns_binding(rt, s, dir_ns.as_deref());
                        let q = quote_form(rt, full);
                        let ge = get_with_default(rt, local, q);
                        binds.push(local);
                        binds.push(ge);
                    }
                }
            } else if is_keyword(rt, key, "as") {
                // {:as whole} -> whole bound to the entire map.
                binds.push(val);
                binds.push(t);
            } else if is_keyword(rt, key, "or") {
                // handled above
            } else {
                // {name :key} -> name (get t :key)  (honoring :or when name is a symbol)
                if matches!(rt.decode(key), Val::Sym(_)) {
                    let ge = get_with_default(rt, key, val);
                    binds.push(key);
                    binds.push(ge);
                } else {
                    let ge = call2(rt, "get", t, val);
                    binds.extend(destructure(rt, key, ge));
                }
            }
            k += 2;
        }
        return binds;
    }
    vec![pat, init]
}

// ─────────────────────────────────────────────────────────────────────────
// Protocols, on the toolkit's defmethod/dispatch axis. A protocol method
// dispatches on `type-of` the first arg (a record's tag: 'Vector/'Rect/…).
// ─────────────────────────────────────────────────────────────────────────

/// `(defprotocol P (m1 [this]) (m2 [this x]))` -> register each method NAME so
/// `(m1 x)` becomes a dispatch site (a sentinel `defmethod` per method; the real
/// impls come from `extend-type`). No-impl calls then error cleanly. Each method
/// is ALSO def'd as a first-class fn wrapping a `%dispatch` site, so methods
/// work in value position (`(reduce prepend …)`), as in Clojure, where protocol
/// methods are real vars.
fn desugar_defprotocol<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let dok = sym(rt, "do");
    let mut out = vec![dok];
    let sentinel = sym(rt, "-protocol-default");
    let mut method_syms = Vec::new();
    for &spec in &items[2..] {
        // Skip a leading docstring / options map — only `(method [params] …)`
        // method specs are lists.
        if rt.as_cons(spec).is_none() {
            continue;
        }
        let parts = rt.list_to_vec(spec);
        let m = parts[0];
        let params = parts[1]; // [this ...] vector record
        let nilv = rt.encode(Val::Nil);
        let fnf = mk_fn(rt, params, vec![nilv]);
        out.push(mk_defmethod(rt, m, sentinel, fnf));
        // (def m (fn ([g…] (%dispatch m g…)) …)) — one clause per declared arity.
        let arity_counts: Vec<usize> = parts[1..]
            .iter()
            .filter_map(|&p| binding_items(rt, p).map(|v| v.len()))
            .collect();
        if !arity_counts.is_empty() {
            let dsym = sym(rt, "%dispatch");
            let mut clauses = Vec::new();
            for n in arity_counts {
                let gs: Vec<u64> = (0..n).map(|_| gensym(rt, "mp")).collect();
                let pv = make_vector(rt, gs.clone());
                let mut call = vec![dsym, m];
                call.extend(gs);
                let body = rt.vec_to_list(&call);
                clauses.push(rt.vec_to_list(&[pv, body]));
            }
            let fnk = sym(rt, "fn");
            let mut fnform = vec![fnk];
            fnform.extend(clauses);
            let wrapper = rt.vec_to_list(&fnform);
            let defk = sym(rt, "def");
            out.push(rt.vec_to_list(&[defk, m, wrapper]));
        }
        // record the RESOLVED (qualified) method name — the same sym the dispatch
        // registry is keyed by — so %method-types matches at reflection time.
        let resolved = match rt.decode(m) {
            Val::Sym(s) => rt.encode(Val::Sym(comp.resolve_ref(rt, s))),
            _ => m,
        };
        let quote = sym(rt, "quote");
        method_syms.push(rt.vec_to_list(&[quote, resolved]));
    }
    // Bind the protocol name to a `(record 'Protocol 'Name (list 'm1 'm2 ...))`
    // so `satisfies?`/`extends?`/`extenders` can reflect over its methods.
    let pname = items[1];
    let quote = sym(rt, "quote");
    let qpname = rt.vec_to_list(&[quote, pname]);
    let listk = sym(rt, "list");
    let mut mlist = vec![listk];
    mlist.extend(method_syms);
    let methods_list = rt.vec_to_list(&mlist);
    let recordk = sym(rt, "record");
    let ptag = {
        let q = sym(rt, "quote");
        let pt = sym(rt, "Protocol");
        rt.vec_to_list(&[q, pt])
    };
    let precord = rt.vec_to_list(&[recordk, ptag, qpname, methods_list]);
    let defk = sym(rt, "def");
    out.push(rt.vec_to_list(&[defk, pname, precord]));
    rt.vec_to_list(&out)
}

/// `(extend-type T P (m1 [this] body) (m2 [this x] body) [P2 ...])` -> a
/// `(defmethod m T (fn [this ...] body))` per method. Protocol NAMES (bare
/// symbols) group the methods; each that resolves to a defined protocol var is
/// also registered as a (possibly marker) extension of T.
fn desugar_extend_type<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    // `(extend-type nil …)`: `nil` reads as the value Nil, but `type-of nil`
    // reports the tag symbol `nil` — so extend against that symbol.
    let tys: Vec<u64> = if matches!(rt.decode(items[1]), Val::Nil) {
        vec![sym(rt, "nil")]
    } else if let Val::Sym(s) = rt.decode(items[1]) {
        // A qualified Java class name (`clojure.lang.IPersistentVector`,
        // `java.lang.String`) maps to our runtime type tag(s) through the JVM
        // layer's registry: `:extend-tags` when the interface covers several
        // concrete types (Named = Symbol + Keyword; IPersistentCollection =
        // every collection tag), else its `:tag`. Method registration needs
        // tags at COMPILE time, so this reads the in-language registry atom
        // from Rust (the policy stays in the data).
        let name = rt.sym_name(s).to_string();
        // A dotted name is host-class-like as-is; a BARE name may be an
        // imported class (`(:import (clojure.lang IPersistentMap))` + bare
        // `extend-protocol` targets — nrepl.bencode's shape). Unresolvable
        // names stay bare dialect tags (core's `(extend-type Vector …)`).
        let fqn = if name.contains('.') {
            Some(name.replace('/', "."))
        } else {
            comp.resolve_class(&name)
        };
        if let Some(fqn) = fqn {
            let tags = jvm_registry_tags(rt, &fqn);
            if tags.is_empty() {
                let simple = name.rsplit(['.', '/']).next().unwrap_or(&name).to_string();
                let t = rt.intern(&simple);
                vec![rt.encode(Val::Sym(t))]
            } else {
                tags.into_iter().map(|t| rt.encode(Val::Sym(t))).collect()
            }
        } else {
            vec![items[1]]
        }
    } else {
        vec![items[1]]
    };
    let dok = sym(rt, "do");
    let mut out = vec![dok];
    for &item in &items[2..] {
        if matches!(rt.decode(item), Val::Sym(_)) {
            // a protocol name — group marker; register the extension (markers
            // have no methods, so satisfies? needs the explicit record).
            for &ty in &tys {
                if let Some(reg) = marker_registration(rt, comp, item, ty) {
                    out.push(reg);
                }
            }
            continue;
        }
        let parts = rt.list_to_vec(item); // (m [this ...] body...)
        let m = parts[0];
        let params = parts[1];
        for &ty in &tys {
            let fnf = mk_fn(rt, params, parts[2..].to_vec());
            out.push(mk_defmethod(rt, m, ty, fnf));
        }
    }
    rt.vec_to_list(&out)
}

/// `(-register-marker P 'T)` when `group` (a protocol-group symbol inside
/// `deftype`/`extend-type`) resolves to a DEFINED var — the runtime guard in
/// `-register-marker` keeps non-protocol values (deftype tag syms) out of the
/// registry. `None` for unresolved names (host interfaces we don't model).
fn marker_registration<M: ValueModel>(
    rt: &mut Runtime<M>,
    comp: &Compiler,
    group: u64,
    ty: u64,
) -> Option<u64> {
    let Val::Sym(s) = rt.decode(group) else { return None };
    let q = comp.resolve_ref(rt, s);
    if !rt.global_defined(q) {
        return None;
    }
    let reg = sym(rt, "-register-marker");
    let pref = rt.encode(Val::Sym(q));
    let tq = quote_form(rt, ty);
    Some(rt.vec_to_list(&[reg, pref, tq]))
}

/// `(deftype T [f0 f1] Protocol (-m [this a] body) …)` — the full ClojureScript
/// `deftype` form. Emits `(def ->T (fn [f0 f1] (record 'T f0 f1)))` PLUS an inline
/// `-proto-method` per protocol method (type-indexed dispatch, same axis as
/// `extend-type`). Inside a method body the deftype's FIELD NAMES are in lexical
/// scope (bound from the instance = first param), matching cljs; multi-arity
/// methods (the same name repeated) become a multi-arity fn. The `Object` group
/// (host interop: toString/equiv/indexOf/…) is skipped. `^:mutable` is ignored
/// (the reader already drops the meta) — `set!` on such a "field" hits the local
/// binding, so caching-hash degrades to recomputation rather than erroring.
fn desugar_deftype<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let tsym = items[1];
    let tname = match rt.decode(tsym) {
        Val::Sym(s) => rt.sym_name(s).to_string(),
        _ => panic!("deftype: name must be a symbol"),
    };
    let fields = binding_items(rt, items[2]).unwrap_or_else(|| rt.list_to_vec(items[2]));

    // constructor: (def ->T (fn [fields] (record 'T fields)))
    let recsym = sym(rt, "record");
    let tag = quote_form(rt, tsym);
    let mut rec = vec![recsym, tag];
    rec.extend_from_slice(&fields);
    let reccall = rt.vec_to_list(&rec);
    let paramvec = make_vector(rt, fields.clone());
    let ctorfn = mk_fn(rt, paramvec, vec![reccall]);
    let ctorname = sym(rt, &format!("->{tname}"));
    let defk = sym(rt, "def");
    let ctordef = rt.vec_to_list(&[defk, ctorname, ctorfn]);

    // register the field-name order so `(.-field x)` resolves by name.
    let regsym = sym(rt, "%register-fields");
    let tq = quote_form(rt, tsym);
    let flist = rt.vec_to_list(&fields);
    let fq = quote_form(rt, flist);
    let regcall = rt.vec_to_list(&[regsym, tq, fq]);

    let dok = sym(rt, "do");
    let mut out = vec![dok, ctordef, regcall];
    // Bind the bare type name to its type symbol, so `T` resolves as a value — used
    // as a dispatch value (`(defmethod print-method T …)`) or in bare references.
    let name_q = quote_form(rt, tsym);
    out.push(rt.vec_to_list(&[defk, tsym, name_q]));

    // inline protocol methods: a bare symbol starts a protocol group (or `Object`);
    // lists are method impls, and consecutive same-name impls are one multi-arity fn.
    let mut i = 3;
    let mut in_object = false;
    while i < items.len() {
        let it = items[i];
        if matches!(rt.decode(it), Val::Sym(_)) {
            in_object = is_sym(rt, it, "Object");
            if !in_object {
                // Register T as extending this protocol — markers (method-less
                // interfaces like core.match's IPseudoPattern) are only
                // satisfiable through this record.
                if let Some(reg) = marker_registration(rt, comp, it, tsym) {
                    out.push(reg);
                }
            }
            i += 1;
            continue;
        }
        let mname = rt.list_to_vec(it)[0];
        // gather this + following consecutive impls sharing the method name.
        let mut arities: Vec<(u64, Vec<u64>)> = Vec::new();
        loop {
            let p = rt.list_to_vec(items[i]);
            arities.push((p[1], p[2..].to_vec()));
            i += 1;
            if i >= items.len() || matches!(rt.decode(items[i]), Val::Sym(_)) {
                break;
            }
            if !same_sym(rt, rt.list_to_vec(items[i])[0], mname) {
                break;
            }
        }
        if in_object {
            continue; // host interop methods — not dispatched
        }
        // Translate a JVM `clojure.lang.*` interface method to our cljs-style
        // protocol method (name + the arity that matches our signature), so a
        // JVM-Clojure deftype (e.g. core.match's) participates in get/nth/seq/etc.
        let mname_str = match rt.decode(mname) {
            Val::Sym(s) => rt.sym_name(s).to_string(),
            _ => String::new(),
        };
        let (target, use_arities) = match xlate_iface_method(&mname_str) {
            Some((tgt, want)) => {
                let filtered: Vec<(u64, Vec<u64>)> = arities
                    .iter()
                    .filter(|(pv, _)| binding_items(rt, *pv).map(|v| v.len()).unwrap_or(0) == want)
                    .cloned()
                    .collect();
                (sym(rt, tgt), if filtered.is_empty() { arities } else { filtered })
            }
            None => (mname, arities),
        };
        let mfn = build_method_fn(rt, &fields, &use_arities);
        out.push(mk_defmethod(rt, target, tsym, mfn));
    }
    rt.vec_to_list(&out)
}

/// Map a JVM `clojure.lang.*` interface method name to our protocol method name
/// and the TOTAL param count (incl. `this`) of the arity to keep — so a deftype
/// implementing the Java interfaces works with our cljs-style protocols.
fn xlate_iface_method(name: &str) -> Option<(&'static str, usize)> {
    Some(match name {
        "valAt" => ("-lookup", 3),        // (valAt [this k nf])
        "nth" => ("-nth", 2),             // (nth [this i])
        "first" => ("-first", 1),
        "more" | "rest" | "next" => ("-rest", 1),
        "seq" => ("-seq", 1),
        "count" => ("-count", 1),
        "cons" | "conj" => ("-conj", 2),
        "assoc" => ("-assoc", 3),
        "containsKey" => ("-contains-key?", 2),
        "empty" => ("-empty", 1),
        "peek" => ("-peek", 1),
        "pop" => ("-pop", 1),
        "invoke" => ("-invoke", 2), // 1-arg IFn: (invoke [this a])
        _ => return None,
    })
}

fn same_sym<M: ValueModel>(rt: &Runtime<M>, a: u64, b: u64) -> bool {
    matches!((rt.decode(a), rt.decode(b)), (Val::Sym(x), Val::Sym(y)) if x == y)
}

/// Build a method fn (single- or multi-arity) whose body has the deftype's fields
/// bound from the instance (first param), except fields shadowed by a param name.
fn build_method_fn<M: ValueModel>(
    rt: &mut Runtime<M>,
    fields: &[u64],
    arities: &[(u64, Vec<u64>)],
) -> u64 {
    if arities.len() == 1 {
        let (params, body) = &arities[0];
        let wrapped = wrap_field_let(rt, fields, *params, body);
        return mk_fn(rt, *params, vec![wrapped]);
    }
    let fnk = sym(rt, "fn");
    let mut out = vec![fnk];
    for (params, body) in arities {
        let wrapped = wrap_field_let(rt, fields, *params, body);
        out.push(rt.vec_to_list(&[*params, wrapped]));
    }
    rt.vec_to_list(&out)
}

/// `(let [f0 (field inst 0) f1 (field inst 1) …] body…)` — binds each deftype
/// field (that a param doesn't shadow) to its slot on the instance.
fn wrap_field_let<M: ValueModel>(
    rt: &mut Runtime<M>,
    fields: &[u64],
    params: u64,
    body: &[u64],
) -> u64 {
    let plist = binding_items(rt, params).unwrap_or_else(|| rt.list_to_vec(params));
    if plist.is_empty() {
        let dok = sym(rt, "do");
        let mut out = vec![dok];
        out.extend_from_slice(body);
        return rt.vec_to_list(&out);
    }
    let inst = plist[0];
    let param_syms: std::collections::HashSet<Sym> = plist
        .iter()
        .filter_map(|&p| match rt.decode(p) {
            Val::Sym(s) => Some(s),
            _ => None,
        })
        .collect();
    let fieldk = sym(rt, "field");
    let mut binds = Vec::new();
    for (idx, &f) in fields.iter().enumerate() {
        if let Val::Sym(s) = rt.decode(f) {
            if param_syms.contains(&s) {
                continue; // a param shadows this field
            }
        }
        let idxv = rt.encode(Val::Int(idx as i128));
        let fieldcall = rt.vec_to_list(&[fieldk, inst, idxv]);
        binds.push(f);
        binds.push(fieldcall);
    }
    if binds.is_empty() {
        let dok = sym(rt, "do");
        let mut out = vec![dok];
        out.extend_from_slice(body);
        return rt.vec_to_list(&out);
    }
    let bindvec = make_vector(rt, binds);
    let letk = sym(rt, "let");
    let mut out = vec![letk, bindvec];
    out.extend_from_slice(body);
    rt.vec_to_list(&out)
}

fn mk_fn<M: ValueModel>(rt: &mut Runtime<M>, params: u64, body: Vec<u64>) -> u64 {
    let fnk = sym(rt, "fn");
    let mut out = vec![fnk, params];
    out.extend(body);
    rt.vec_to_list(&out)
}

fn mk_defmethod<M: ValueModel>(rt: &mut Runtime<M>, m: u64, ty: u64, imp: u64) -> u64 {
    // Internal protocol-impl form (type-indexed dispatch). Named distinctly from
    // the user-facing `defmethod` (multimethods), which is a library macro.
    let dm = sym(rt, "-proto-method");
    rt.vec_to_list(&[dm, m, ty, imp])
}

/// `(loop [n init ...] body)` -> a self-recursive fn bound via `set!` (so `recur`
/// can tail-jump), then called. `recur` gets TCO from the toolkit trampoline.
///   (let [g nil] (set! g (fn [n ...] body[recur->g])) (g init ...))
fn expand_loop<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let binds = match binding_items(rt, items[1]) {
        Some(l) => l,
        None => rt.list_to_vec(items[1]),
    };
    let mut names = Vec::new();
    let mut inits = Vec::new();
    let mut i = 0;
    while i + 1 < binds.len() {
        names.push(binds[i]);
        inits.push(binds[i + 1]);
        i += 2;
    }
    let g = gensym(rt, "loop");
    // Self-parameterized loop: the loop fn takes ITSELF as its first param, so
    // `recur` becomes `(g g …)` where the callee `g` is a param (up:0), not an
    // outer local. That keeps the fn body from referencing any parent frame,
    // which is exactly the condition the JIT needs to lift the loop into
    // registers — an in-place native loop, the same fast path a self-recursive
    // `defn` gets (~3× faster than the old outer-local closure). It also lets us
    // drop the `set!`/nil self-box entirely (the fn no longer forward-references
    // its own binding).
    let body: Vec<u64> = items[2..].iter().map(|&b| replace_recur(rt, b, g)).collect();

    // (fn [g name0 name1 …] body) — `g` (self) is the first param.
    let fnk = sym(rt, "fn");
    let mut params = vec![g];
    params.extend(names.iter().copied());
    let paramvec = make_vector(rt, params);
    let mut fnform = vec![fnk, paramvec];
    fnform.extend(body);
    let fnform = rt.vec_to_list(&fnform);

    // (let [g fnform]
    //   (let [name0 init0 name1 init1 …] (g g name0 name1 …)))
    // The inner `let` binds the loop names to their inits SEQUENTIALLY (so a later
    // init can see an earlier binding, matching `let`/Clojure), then kicks off the
    // loop by calling `g` with ITSELF followed by the initial values.
    let letk = sym(rt, "let");
    let gbindvec = make_vector(rt, vec![g, fnform]);
    let mut initbinds = Vec::new();
    for (n, ini) in names.iter().zip(inits.iter()) {
        initbinds.push(*n);
        initbinds.push(*ini);
    }
    let initbindvec = make_vector(rt, initbinds);
    let mut gcall = vec![g, g];
    gcall.extend(names.iter().copied());
    let gcallform = rt.vec_to_list(&gcall);
    let initlet = rt.vec_to_list(&[letk, initbindvec, gcallform]);
    let letform = rt.vec_to_list(&[letk, gbindvec, initlet]);
    expand(rt, cs, macros, comp, letform)
}

/// Replace `(recur ...)` with `(g g ...)` — a self-tail-call to the loop fn (which
/// takes itself as its first param). Does not descend into a nested `fn`/`loop`
/// (which rebinds the recur target).
fn replace_recur<M: ValueModel>(rt: &mut Runtime<M>, form: u64, g: u64) -> u64 {
    if let Some((h, _)) = rt.as_cons(form) {
        if is_sym(rt, h, "recur") {
            let items = rt.list_to_vec(form);
            let mut out = vec![g, g];
            out.extend_from_slice(&items[1..]);
            return rt.vec_to_list(&out);
        }
        if is_sym(rt, h, "fn") || is_sym(rt, h, "loop") || is_sym(rt, h, "quote") {
            return form;
        }
        let items = rt.list_to_vec(form);
        let out: Vec<u64> = items.iter().map(|&it| replace_recur(rt, it, g)).collect();
        return rt.vec_to_list(&out);
    }
    form
}

/// `(fn* name? [params] body)` or `(fn* name? ([p] b) ([p q] b) …)`: drop the
/// optional self-name, then either normalize to a single-arity `fn*` (binder
/// handles destructuring) or desugar multiple arities to one variadic fn that
/// dispatches on `(count args)`.
fn expand_fn<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let i = 1;
    if i < items.len() && matches!(rt.decode(items[i]), Val::Sym(_)) {
        // Named fn `(fn name . clauses)`: `name` must be in scope inside the body,
        // bound to the closure itself (letrec semantics — supports general, not
        // just tail, recursion). Desugar via a cell that holds the closure:
        //   (let [-box (%atom-new nil)]
        //     (%atom-set -box (fn . clauses'))    ; clauses' wrap each body with
        //     (%atom-get -box))                   ;   (let [name (%atom-get -box)] ..)
        // so `name` derefs the closure at CALL time (the atom is filled by then).
        let name = items[i];
        let box_sym = gensym(rt, "self-box");
        let rest = &items[i + 1..];
        // Wrap the clause bodies so `name` resolves to `(%atom-get box)`.
        let cellref = {
            let cr = sym(rt, "%atom-get");
            rt.vec_to_list(&[cr, box_sym])
        };
        let wrap_body = |rt: &mut Runtime<M>, body: &[u64]| -> u64 {
            let letk = sym(rt, "let");
            let bvec = make_vector(rt, vec![name, cellref]);
            let mut lf = vec![letk, bvec];
            lf.extend_from_slice(body);
            rt.vec_to_list(&lf)
        };
        // single-arity iff rest[0] is a params container (vector / legacy sym-list)
        // or there is no clause at all.
        let single = rest.is_empty()
            || binding_items(rt, rest[0]).is_some()
            || rt.as_cons(rest[0]).map(|(h, _)| matches!(rt.decode(h), Val::Sym(_))).unwrap_or(true);
        let fnk = sym(rt, "fn");
        let inner_fn = if single {
            let params = if rest.is_empty() { make_vector(rt, vec![]) } else { rest[0] };
            let wrapped = wrap_body(rt, if rest.is_empty() { &[] } else { &rest[1..] });
            rt.vec_to_list(&[fnk, params, wrapped])
        } else {
            let mut clauses = vec![fnk];
            for &cl in rest {
                let cv = rt.list_to_vec(cl);
                let params = cv[0];
                let wrapped = wrap_body(rt, &cv[1..]);
                clauses.push(rt.vec_to_list(&[params, wrapped]));
            }
            rt.vec_to_list(&clauses)
        };
        let cellk = sym(rt, "%atom-new");
        let niln = rt.encode(Val::Nil);
        let cellnew = rt.vec_to_list(&[cellk, niln]);
        let setk = sym(rt, "%atom-set");
        let setf = rt.vec_to_list(&[setk, box_sym, inner_fn]);
        let letk = sym(rt, "let");
        let bvec = make_vector(rt, vec![box_sym, cellnew]);
        let cellref2 = {
            let cr = sym(rt, "%atom-get");
            rt.vec_to_list(&[cr, box_sym])
        };
        let letform = rt.vec_to_list(&[letk, bvec, setf, cellref2]);
        return expand(rt, cs, macros, comp, letform);
    }
    // Single-arity: params are a `[..]` vector, a `(a b)` symbol-list (our legacy
    // style), or empty. Multi-arity: each element is a CLAUSE `([params] body)`
    // whose first element is a params container.
    let single = if i >= items.len() {
        true
    } else if binding_items(rt, items[i]).is_some() {
        true
    } else if let Some((h, _)) = rt.as_cons(items[i]) {
        matches!(rt.decode(h), Val::Sym(_))
    } else {
        true
    };
    if single {
        let params = items[i];
        let (params, body) = wrap_fn_recur(rt, params, &items[i + 1..]);
        let fnstar = sym(rt, "fn*");
        let mut norm = vec![fnstar, params];
        norm.extend(body);
        let normf = rt.vec_to_list(&norm);
        rebuild_binder(rt, cs, macros, comp, normf)
    } else {
        let clauses: Vec<u64> = items[i..].to_vec();
        let desugared = multi_arity(rt, &clauses);
        expand(rt, cs, macros, comp, desugared)
    }
}

/// A `fn` body that contains `recur` re-enters the fn — desugar to a `loop` over
/// the params (so `recur` rebinds them and tail-jumps via the loop trampoline).
/// Returns `(new_params, new_body)`: a DESTRUCTURING param (e.g. `[x :as xs]`) is
/// replaced by a fresh gensym in the param list — so the loop binds every position
/// (matching `recur`'s arity) — and the pattern is destructured from that gensym
/// via a `let` inside the loop, re-run each iteration. When there's no `recur`,
/// params and body pass through unchanged.
fn wrap_fn_recur<M: ValueModel>(rt: &mut Runtime<M>, params: u64, body: &[u64]) -> (u64, Vec<u64>) {
    let plist = match binding_items(rt, params) {
        Some(l) => l,
        None => rt.list_to_vec(params),
    };
    if plist.is_empty() || !body.iter().any(|&b| has_recur(rt, b)) {
        return (params, body.to_vec());
    }
    let mut new_params = Vec::new();
    let mut loop_binds = Vec::new();
    let mut destr = Vec::new(); // pattern, gensym pairs to destructure inside the loop
    let mut amp = false;
    for &p in &plist {
        if matches!(rt.decode(p), Val::Sym(_)) {
            new_params.push(p);
            if is_sym(rt, p, "&") {
                amp = true;
                continue;
            }
            // a normal symbol, or the rest name after `&`: loop-bind it to itself
            loop_binds.push(p);
            loop_binds.push(p);
            amp = false;
        } else {
            // a destructuring pattern (possibly the variadic rest pattern): gensym it
            let g = gensym(rt, "p");
            new_params.push(g);
            loop_binds.push(g);
            loop_binds.push(g);
            destr.push(p);
            destr.push(g);
            amp = false;
        }
    }
    let _ = amp;
    let inner: Vec<u64> = if destr.is_empty() {
        body.to_vec()
    } else {
        let letk = sym(rt, "let");
        let bvec = make_vector(rt, destr);
        let mut letf = vec![letk, bvec];
        letf.extend_from_slice(body);
        vec![rt.vec_to_list(&letf)]
    };
    let loopsym = sym(rt, "loop");
    let bindvec = make_vector(rt, loop_binds);
    let mut loopf = vec![loopsym, bindvec];
    loopf.extend(inner);
    let new_params = make_vector(rt, new_params);
    (new_params, vec![rt.vec_to_list(&loopf)])
}

/// Does `ir` contain a `(recur …)` in this fn's scope (not inside a nested
/// `fn`/`loop`, which rebind the recur target)?
fn has_recur<M: ValueModel>(rt: &Runtime<M>, form: u64) -> bool {
    let Some((h, _)) = rt.as_cons(form) else { return false };
    if is_sym(rt, h, "recur") {
        return true;
    }
    if is_sym(rt, h, "fn") || is_sym(rt, h, "fn*") || is_sym(rt, h, "loop") || is_sym(rt, h, "quote") {
        return false;
    }
    rt.list_to_vec(form).iter().any(|&c| has_recur(rt, c))
}

/// Multiple arities -> `(fn* [& g] (let [n (count g)] (cond (= n k0) (let [pv0 g] b0) … true (let [pvV g] bV))))`.
/// Each arity's params destructure `g`; a variadic clause is the catch-all.
fn multi_arity<M: ValueModel>(rt: &mut Runtime<M>, clauses: &[u64]) -> u64 {
    // Per-arity BODIES, selected by argument count in the runtime (`%multifn`)
    // — real Clojure's IFn.invoke(a, b, …) overloads. Each fixed-arity clause
    // becomes an ordinary fixed-arity fn (register-callable on the JIT,
    // inlinable, no rest-list allocation); the variadic clause stays a
    // variadic fn serving every higher count.
    //   (fn ([a] e1) ([a b] e2) ([a b & r] e3))
    //   => (%multifn (fn* [a] e1') (fn* [a b] e2') (fn* [a b & r] e3'))
    let fnstar = sym(rt, "fn*");
    let mf = sym(rt, "%multifn");
    let mut out = vec![mf];
    for &clause in clauses {
        let parts = rt.list_to_vec(clause);
        let (paramvec, body) = wrap_fn_recur(rt, parts[0], &parts[1..]);
        let mut f = vec![fnstar, paramvec];
        f.extend(body);
        out.push(rt.vec_to_list(&f));
    }
    rt.vec_to_list(&out)
}

// ─────────────────────────────────────────────────────────────────────────
// Java-interop lowering — SYNTAX only. The expander knows no class or method
// names: instance calls become dispatch sites on dot-munged method names
// (registered by `defclass`, same inline-cached machinery as protocols), and
// statics/constructors/class values become calls into the in-language JVM
// layer's registry (see host_jvm_src). Misses are catchable runtime errors.
// ─────────────────────────────────────────────────────────────────────────

/// `(%dispatch .method obj args…)` — an instance-method call site.
fn dot_dispatch<M: ValueModel>(rt: &mut Runtime<M>, method: &str, obj: u64, args: &[u64]) -> u64 {
    let d = sym(rt, "%dispatch");
    let m = sym(rt, &format!(".{method}"));
    let mut out = vec![d, m, obj];
    out.extend_from_slice(args);
    rt.vec_to_list(&out)
}

/// `(f 'a 'b args…)` — a call to a JVM-layer fn with leading quoted symbols.
fn jvm_layer_call<M: ValueModel>(
    rt: &mut Runtime<M>,
    f: &str,
    syms: &[&str],
    args: &[u64],
) -> u64 {
    let fs = sym(rt, f);
    let mut out = vec![fs];
    for s in syms {
        let sv = sym(rt, s);
        out.push(quote_form(rt, sv));
    }
    out.extend_from_slice(args);
    rt.vec_to_list(&out)
}

/// If `target` (the first arg of a `(. target …)` form) names a CLASS — a
/// dotted name with a capitalized last segment, or a bare imported simple
/// name — its fully-qualified name. `None` = an instance expression.
fn static_class_fqn<M: ValueModel>(rt: &Runtime<M>, comp: &Compiler, target: u64) -> Option<String> {
    let Val::Sym(s) = rt.decode(target) else { return None };
    let name = rt.sym_name(s);
    let simple = last_seg(name);
    if !simple.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return None;
    }
    if name.contains('.') {
        return Some(name.to_string());
    }
    comp.resolve_class(&simple)
}

fn interop_rewrite<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> Option<u64> {
    let items = rt.list_to_vec(form);
    let Val::Sym(hs) = rt.decode(items[0]) else { return None };
    let hname = rt.sym_name(hs).to_string();

    if hname == "." && items.len() >= 3 {
        // `(. target member)` / `(. target member args…)` / `(. target (member args…))`
        let (method, margs): (String, Vec<u64>) = if rt.as_cons(items[2]).is_some() {
            let call = rt.list_to_vec(items[2]);
            (sym_str(rt, call[0]), call[1..].to_vec())
        } else {
            (sym_str(rt, items[2]), items[3..].to_vec())
        };
        return Some(match static_class_fqn(rt, comp, items[1]) {
            Some(fqn) => {
                if margs.is_empty() && rt.as_cons(items[2]).is_none() {
                    // `(. Class member)` — a static field read (or 0-arg static fn)
                    jvm_layer_call(rt, "-jvm-static-member", &[&fqn, &method], &[])
                } else {
                    jvm_layer_call(rt, "-jvm-invoke-static", &[&fqn, &method], &margs)
                }
            }
            None => dot_dispatch(rt, &method, items[1], &margs),
        });
    }
    if let Some(slash) = hname.find('/') {
        // `Foo/bar` / `pkg.Foo/bar` with a capitalized (non-alias) class segment
        // is a static call; a lowercase prefix is a ns/alias var reference.
        let left = &hname[..slash];
        let simple = last_seg(left);
        if simple.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && !comp.has_alias(left) {
            let fqn = if left.contains('.') {
                left.to_string()
            } else {
                comp.resolve_class(&simple).unwrap_or_else(|| simple.clone())
            };
            let method = &hname[slash + 1..];
            return Some(jvm_layer_call(rt, "-jvm-invoke-static", &[&fqn, method], &items[1..]));
        }
    }
    // `(.-field x)` -> `(%field-by-name x 'field)` — ClojureScript field access on
    // a deftype instance, resolved through the field-name registry. NOT part of
    // the JVM layer: these are the dialect's own record fields.
    if let Some(field) = hname.strip_prefix(".-") {
        if !field.is_empty() {
            // `.-length` (a cljs raw array's element count) -> `(%alength x)`.
            if field == "length" {
                return Some(call1(rt, "%alength", items[1]));
            }
            let fsym = sym(rt, field);
            let fq = quote_form(rt, fsym);
            let fbn = sym(rt, "%field-by-name");
            return Some(rt.vec_to_list(&[fbn, items[1], fq]));
        }
    }
    if hname.starts_with('.') && hname.len() > 1 {
        return Some(dot_dispatch(rt, &hname[1..], items[1], &items[2..]));
    }
    // Constructor `(Class. args…)`. A known class (dotted, `ns/Class`-qualified,
    // or imported) constructs through the registry; a bare unknown name is a
    // deftype/dialect record — build it directly (`(PersistentVector. …)` is the
    // collection hot path, so no registry indirection).
    if hname.ends_with('.') && hname.len() > 1 && hname != ".." {
        let cname = &hname[..hname.len() - 1];
        let simple = last_seg(cname);
        let fqn = if cname.contains('.') || cname.contains('/') {
            Some(cname.replace('/', "."))
        } else {
            comp.resolve_class(&simple)
        };
        if let Some(fqn) = fqn {
            return Some(jvm_layer_call(rt, "-jvm-construct", &[&fqn], &items[1..]));
        }
        let rec = sym(rt, "record");
        let tag = sym(rt, &simple);
        let tag_q = quote_form(rt, tag);
        let mut out = vec![rec, tag_q];
        out.extend_from_slice(&items[1..]);
        return Some(rt.vec_to_list(&out));
    }
    None
}

fn sym_str<M: ValueModel>(rt: &Runtime<M>, v: u64) -> String {
    match rt.decode(v) {
        Val::Sym(s) => rt.sym_name(s).to_string(),
        _ => String::new(),
    }
}
fn last_seg(s: &str) -> String {
    // the simple class name: the segment after the last `.` or `/`
    // (e.g. `cljs.core/MapEntry` -> `MapEntry`).
    s.rsplit(['.', '/']).next().unwrap_or(s).to_string()
}

/// The runtime tag(s) an `extend-type` on a registered host class should
/// register methods against, read from the IN-LANGUAGE JVM registry
/// (`clojure.core/-jvm-registry`: an atom holding a flat `(fqn desc …)` plist
/// of `JvmClass` descriptor records; see clj/host_jvm.clj). Prefers the
/// descriptor's `:extend-tags` list (field 11 — an interface spanning several
/// concrete types), else its single `:tag` (field 2). Empty = unregistered.
/// The expander needs this at compile time, but the knowledge itself stays in
/// the language — Rust just walks the data.
fn jvm_registry_tags<M: ValueModel>(rt: &Runtime<M>, fqn: &str) -> Vec<Sym> {
    let regsym = rt.intern("clojure.core/-jvm-registry");
    let Some(regv) = rt.global(regsym) else { return vec![] };
    let Val::Ref(id) = rt.decode(regv) else { return vec![] };
    let ObjView::Atom(a) = rt.view_gc(id) else { return vec![] };
    let plist = rt.list_to_vec(a.load(Ordering::Acquire));
    let want = rt.intern(fqn);
    let Some(desc) = plist
        .chunks(2)
        .find(|kv| kv.len() == 2 && matches!(rt.decode(kv[0]), Val::Sym(s) if s == want))
        .map(|kv| kv[1])
    else {
        return vec![];
    };
    let Val::Ref(did) = rt.decode(desc) else { return vec![] };
    let ObjView::Record { fields, .. } = rt.view_gc(did) else { return vec![] };
    let (tag, ext) = (fields.get(2).copied(), fields.get(11).copied());
    if let Some(ext) = ext {
        if !matches!(rt.decode(ext), Val::Nil) {
            return rt
                .list_to_vec(ext)
                .into_iter()
                .filter_map(|t| match rt.decode(t) {
                    Val::Sym(s) => Some(s),
                    _ => None,
                })
                .collect();
        }
    }
    match tag.map(|t| rt.decode(t)) {
        Some(Val::Sym(s)) => vec![s],
        _ => vec![],
    }
}

fn gensym<M: ValueModel>(rt: &mut Runtime<M>, prefix: &str) -> u64 {
    let n = GENSYM.fetch_add(1, Ordering::Relaxed);
    let s = rt.intern(&format!("{prefix}__{n}__d"));
    rt.encode(Val::Sym(s))
}

fn int<M: ValueModel>(rt: &mut Runtime<M>, i: i128) -> u64 {
    rt.encode(Val::Int(i))
}

fn call2<M: ValueModel>(rt: &mut Runtime<M>, f: &str, a: u64, b: u64) -> u64 {
    let fs = sym(rt, f);
    rt.vec_to_list(&[fs, a, b])
}

/// `(record 'Keyword 'name)` — a keyword-construction form for `:name`.
fn keyword_expr<M: ValueModel>(rt: &mut Runtime<M>, name_sym: u64) -> u64 {
    let rec = sym(rt, "record");
    let kw = sym(rt, "Keyword");
    let tag = quote_form(rt, kw);
    let nm = quote_form(rt, name_sym);
    rt.vec_to_list(&[rec, tag, nm])
}

/// A REAL runtime vector (phase-appropriate) — the expander's synthesized
/// binding/param vectors are the same data the reader produces.
fn make_vector<M: ValueModel>(rt: &mut Runtime<M>, elems: Vec<u64>) -> u64 {
    data::make_vector(rt, &elems)
}

/// Is `k` a `:keys`-style directive, and with what namespace? `:keys` -> the
/// bare form (None); `:a/keys` -> Some("a"). Anything else -> not a directive.
fn keys_directive<M: ValueModel>(rt: &Runtime<M>, k: u64) -> Option<Option<String>> {
    ns_directive(rt, k, "keys")
}
/// The `:syms` counterpart of `keys_directive`.
fn syms_directive<M: ValueModel>(rt: &Runtime<M>, k: u64) -> Option<Option<String>> {
    ns_directive(rt, k, "syms")
}
fn ns_directive<M: ValueModel>(rt: &Runtime<M>, k: u64, want: &str) -> Option<Option<String>> {
    let f0 = record_field0(rt, k, reader::KEYWORD)?;
    let Val::Sym(sy) = rt.decode(f0) else { return None };
    let name = rt.sym_name(sy);
    match name.rsplit_once('/') {
        Some((ns, last)) if last == want => Some(Some(ns.to_string())),
        None if name == want => Some(None),
        _ => None,
    }
}

/// Split a destructuring binding symbol into (LOCAL name, FULL key name), per
/// Clojure: the local is always the bare name, and the key carries the
/// namespace — from the symbol itself (`{:keys [ns/x]}`) or, failing that, from
/// the directive (`{:ns/keys [x]}`). A symbol's own namespace wins.
fn split_ns_binding<M: ValueModel>(rt: &mut Runtime<M>, s: u64, dir_ns: Option<&str>) -> (u64, u64) {
    let Val::Sym(sy) = rt.decode(s) else { return (s, s) };
    let name = rt.sym_name(sy).to_string();
    match name.rsplit_once('/') {
        // `ns/x` — bind `x`, look up `:ns/x`
        Some((_, local)) => {
            let l = rt.intern(local);
            (rt.encode(Val::Sym(l)), s)
        }
        None => match dir_ns {
            // `{:ns/keys [x]}` — bind `x`, look up `:ns/x`
            Some(ns) => {
                let f = rt.intern(&format!("{ns}/{name}"));
                (s, rt.encode(Val::Sym(f)))
            }
            None => (s, s),
        },
    }
}

fn is_keyword<M: ValueModel>(rt: &Runtime<M>, k: u64, name: &str) -> bool {
    match record_field0(rt, k, reader::KEYWORD) {
        Some(f0) => is_sym(rt, f0, name),
        None => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Syntax-quote: `template with ~unquote and ~@splice, plus foo# auto-gensym.
// Produces a FORM that builds the templated data at run time.
// ─────────────────────────────────────────────────────────────────────────

fn syntax_quote<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64, gs: &mut HashMap<Sym, Sym>) -> u64 {
    match rt.decode(form) {
        // symbol -> 'symbol, or a per-template gensym for `foo#`
        Val::Sym(s) => {
            let name = rt.sym_name(s).to_string();
            if name.ends_with('#') {
                let g = match gs.get(&s) {
                    Some(&g) => g,
                    None => {
                        let n = GENSYM.fetch_add(1, Ordering::Relaxed);
                        let g = rt.intern(&format!("{}__{}__auto", &name[..name.len() - 1], n));
                        gs.insert(s, g);
                        g
                    }
                };
                let gv = rt.encode(Val::Sym(g));
                quote_form(rt, gv)
            } else {
                // Auto-qualify to the current namespace (macro hygiene), so a
                // macro's helper references resolve at the expansion site.
                let q = comp.qualify_for_syntax_quote(rt, s);
                let qv = rt.encode(Val::Sym(q));
                quote_form(rt, qv)
            }
        }
        // self-evaluating literals stay as themselves
        Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => form,
        Val::Ref(_) => {
            // ~expr -> expr
            if let Some((h, _)) = rt.as_cons(form) {
                if is_sym(rt, h, "unquote") {
                    return rt.list_to_vec(form)[1];
                }
                // a list -> (concat <seq parts>)
                let items = rt.list_to_vec(form);
                return sq_concat(rt, comp, &items, gs);
            }
            // [..] -> (vec (concat ..))
            if let Some(items) = data::vector_items(rt, form) {
                let inner = sq_concat(rt, comp, &items, gs);
                return call1(rt, "vec", inner);
            }
            // {..} -> (hash-map 'k (sq v) ..)  (no splice in map position)
            if let Some(items) = data::map_entries(rt, form) {
                let hm = sym(rt, "hash-map");
                let mut out = vec![hm];
                for &it in &items {
                    out.push(syntax_quote(rt, comp, it, gs));
                }
                return rt.vec_to_list(&out);
            }
            // #{..} -> (apply hash-set (concat ..))  (splices allowed, like lists)
            if let Some(items) = data::set_items(rt, form) {
                let inner = sq_concat(rt, comp, &items, gs);
                let ap = sym(rt, "apply");
                let hs = sym(rt, "hash-set");
                return rt.vec_to_list(&[ap, hs, inner]);
            }
            // keyword / string / char -> themselves (self-evaluating)
            form
        }
    }
}

/// `(-concat (list e0) (list e1) ..)`, with `~@x` contributing `x` directly.
/// Uses the EAGER `-concat` (not the lazy user `concat`): a macro must return a
/// realized code form the expander can splice, not a lazy seq.
fn sq_concat<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, items: &[u64], gs: &mut HashMap<Sym, Sym>) -> u64 {
    let concat = sym(rt, "-concat");
    let mut parts = vec![concat];
    for &it in items {
        if let Some((h, _)) = rt.as_cons(it) {
            if is_sym(rt, h, "unquote-splice") {
                parts.push(rt.list_to_vec(it)[1]);
                continue;
            }
        }
        let one = syntax_quote(rt, comp, it, gs);
        parts.push(call1(rt, "list", one));
    }
    rt.vec_to_list(&parts)
}

fn quote_form<M: ValueModel>(rt: &mut Runtime<M>, x: u64) -> u64 {
    let q = sym(rt, "quote");
    rt.vec_to_list(&[q, x])
}

fn call1<M: ValueModel>(rt: &mut Runtime<M>, f: &str, arg: u64) -> u64 {
    let fs = sym(rt, f);
    rt.vec_to_list(&[fs, arg])
}

/// The elements of a binding/param vector — any runtime vector representation
/// (see `data::vector_items`). Returns `None` if `form` is not a vector.
fn binding_items<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Vec<u64>> {
    data::vector_items(rt, form)
}

fn record_field0<M: ValueModel>(rt: &Runtime<M>, form: u64, tag: &str) -> Option<u64> {
    if let Val::Ref(id) = rt.decode(form) {
        if let ObjView::Record { type_id, fields } = rt.view_gc(id) {
            if rt.sym_name(type_id) == tag {
                return fields.first().copied();
            }
        }
    }
    None
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, n: &str) -> u64 {
    let s = rt.intern(n);
    rt.encode(Val::Sym(s))
}

fn is_sym<M: ValueModel>(rt: &Runtime<M>, bits: u64, name: &str) -> bool {
    matches!(rt.decode(bits), Val::Sym(s) if rt.sym_name(s) == name)
}

/// Strip a leading `(quote X)` wrapper, returning `X` (or the form unchanged).
fn unquote<M: ValueModel>(rt: &Runtime<M>, f: u64) -> u64 {
    if let Some((h, _)) = rt.as_cons(f) {
        if is_sym(rt, h, "quote") {
            return rt.list_to_vec(f)[1];
        }
    }
    f
}

/// If `f` is `(resolve 'x)` or `(ns-resolve 'ns 'x)` with LITERAL quoted symbols,
/// return the fully-qualified sym it resolves to (namespace resolution is a
/// compile-time operation over the current/ named namespace). Else `None`.
fn resolve_rewrite<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, f: u64) -> Option<Sym> {
    let (head, _) = rt.as_cons(f)?;
    let items = rt.list_to_vec(f);
    if is_sym(rt, head, "resolve") && items.len() == 2 {
        // `(resolve 'x)` -> resolve `x` in the CURRENT namespace.
        if let Val::Sym(s) = rt.decode(unquote(rt, items[1])) {
            return Some(comp.resolve_ref(rt, s));
        }
    } else if is_sym(rt, head, "ns-resolve") && items.len() == 3 {
        // `(ns-resolve 'ns 'x)` -> the qualified sym `ns/x` directly.
        let ns = sym_name_of(rt, items[1])?;
        let name = sym_name_of(rt, items[2])?;
        return Some(rt.intern(&format!("{ns}/{name}")));
    }
    None
}

/// The name of a (possibly quoted) symbol form, e.g. `foo.bar` or `'foo.bar`.
fn sym_name_of<M: ValueModel>(rt: &Runtime<M>, f: u64) -> Option<String> {
    match rt.decode(unquote(rt, f)) {
        Val::Sym(s) => Some(rt.sym_name(s).to_string()),
        _ => None,
    }
}

/// Handle a namespace declaration form (`ns`/`in-ns`/`require`/`use`/`alias`/
/// `refer`). These are COMPILE-TIME only: they mutate the compiler's resolution
/// state and produce nil. Returns `None` if `form` isn't such a declaration.
fn handle_ns_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
) -> Option<u64> {
    let (head, _) = rt.as_cons(form)?;
    let nil = rt.encode(Val::Nil);
    if is_sym(rt, head, "ns") {
        let items = rt.list_to_vec(form);
        if let Some(name) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            comp.set_ns(&name);
            // This ns is being defined here (not from a file); don't re-load it.
            comp.mark_loaded(&name);
        }
        // A clause may `require` a namespace, which LOADS AND RUNS a file — a
        // safepoint that relocates every clause still to be processed.
        let clauses = RootVec::new(rt, items.get(2..).unwrap_or(&[]));
        for i in 0..clauses.len() {
            let c = clauses.get(rt, i);
            process_ns_clause(rt, cs, macros, comp, c);
        }
        clauses.release(rt);
        return Some(nil);
    }
    if is_sym(rt, head, "in-ns") {
        let items = rt.list_to_vec(form);
        if let Some(name) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            comp.set_ns(&name);
            comp.mark_loaded(&name);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "require") || is_sym(rt, head, "use") {
        let use_all = is_sym(rt, head, "use");
        let items = rt.list_to_vec(form);
        // Each spec load runs a file (a safepoint) — root the remaining specs.
        let specs = RootVec::new(rt, &items[1..]);
        for i in 0..specs.len() {
            let s = specs.get(rt, i);
            process_require_spec(rt, cs, macros, comp, use_all, s);
        }
        specs.release(rt);
        return Some(nil);
    }
    if is_sym(rt, head, "alias") {
        let items = rt.list_to_vec(form);
        if let (Some(a), Some(real)) =
            (items.get(1).and_then(|&f| sym_name_of(rt, f)), items.get(2).and_then(|&f| sym_name_of(rt, f)))
        {
            comp.add_alias(&a, &real);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "refer") {
        let items = rt.list_to_vec(form);
        if let Some(real) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            refer_names(rt, comp, &items[2..], &real, "only");
        }
        return Some(nil);
    }
    if is_sym(rt, head, "import") {
        let items = rt.list_to_vec(form);
        for &spec in &items[1..] {
            let spec = unquote(rt, spec);
            process_import_spec(rt, comp, spec);
        }
        return Some(nil);
    }
    None
}

/// One import spec: `java.io.File` (a full class name) or `(java.util Date
/// UUID)` / `[java.util Date]` (a package + simple names). Registers simple ->
/// FQN in the current ns's import table; class SEMANTICS come from the
/// in-language registry when the name is used.
fn process_import_spec<M: ValueModel>(rt: &mut Runtime<M>, comp: &mut Compiler, spec: u64) {
    if let Some(full) = sym_name_of(rt, spec) {
        comp.add_import(&last_seg(&full), &full);
        return;
    }
    let parts = binding_items(rt, spec).unwrap_or_else(|| rt.list_to_vec(spec));
    let Some(pkg) = parts.first().and_then(|&p| sym_name_of(rt, p)) else { return };
    for &name in &parts[1..] {
        if let Some(simple) = sym_name_of(rt, name) {
            comp.add_import(&simple, &format!("{pkg}.{simple}"));
        }
    }
}

/// A `(:require …)` / `(:use …)` clause inside an `ns` form.
fn process_ns_clause<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    clause: u64,
) {
    let its = rt.list_to_vec(clause);
    if its.is_empty() {
        return;
    }
    // A require spec LOADS AND RUNS a file — a safepoint. Everything read after
    // one moves: the specs still to be processed, AND `items[0]`, which the
    // `:import` check below re-reads once the require loop has already run.
    let items = RootVec::new(rt, &its);
    if is_keyword(rt, items.get(rt, 0), "require") || is_keyword(rt, items.get(rt, 0), "use") {
        let use_all = is_keyword(rt, items.get(rt, 0), "use");
        for i in 1..items.len() {
            let spec = items.get(rt, i);
            process_require_spec(rt, cs, macros, comp, use_all, spec);
        }
    }
    // `(:import (java.util Date) java.io.File)` — per-ns simple-name -> FQN.
    if is_keyword(rt, items.get(rt, 0), "import") {
        for i in 1..items.len() {
            let spec = unquote(rt, items.get(rt, i));
            process_import_spec(rt, comp, spec);
        }
    }
    // :refer-clojure — core is auto-referred already.
    items.release(rt);
}

/// A single require spec: `foo`, `[foo :as bar]`, or `[foo :refer [x y]]`. LOADS
/// the namespace's file (once) before wiring up any alias/refer.
fn process_require_spec<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    refer_all: bool,
    spec: u64,
) {
    let spec = unquote(rt, spec);
    // bare `(require 'foo)` / `(use 'foo)` — load it; `use` also refers everything.
    if let Some(real) = sym_name_of(rt, spec) {
        let real = normalize_ns(&real);
        ensure_loaded(rt, cs, macros, comp, &real);
        if refer_all {
            comp.refer_all(rt, &real);
        }
        return;
    }
    let Some(elems) = binding_items(rt, spec) else { return };
    let Some(real) = elems.first().and_then(|&f| sym_name_of(rt, f)) else { return };
    let real = normalize_ns(&real);
    // `ensure_loaded` READS AND RUNS the required file — a safepoint that
    // relocates every element of this spec, which the `:as`/`:refer` walk below
    // still reads.
    let elems = RootVec::new(rt, &elems);
    ensure_loaded(rt, cs, macros, comp, &real);
    // `(:use [foo …])` refers all of foo (options like :only aren't honored yet).
    if refer_all {
        comp.refer_all(rt, &real);
    }
    let mut k = 1;
    while k < elems.len() {
        if is_keyword(rt, elems.get(rt, k), "as") && k + 1 < elems.len() {
            if let Some(alias) = sym_name_of(rt, elems.get(rt, k + 1)) {
                comp.add_alias(&alias, &real);
            }
            k += 2;
        // `:refer` and cljs's `:refer-macros` both bring names into scope.
        } else if (is_keyword(rt, elems.get(rt, k), "refer")
            || is_keyword(rt, elems.get(rt, k), "refer-macros"))
            && k + 1 < elems.len()
        {
            let list = elems.get(rt, k + 1);
            refer_names(rt, comp, &[list], &real, "");
            k += 2;
        } else {
            k += 1;
        }
    }
    elems.release(rt);
}

/// Map a ClojureScript-namespace to its bundled `clojure.*` equivalent — this
/// dialect is JVM-free like cljs, so `.cljc`/`.cljs` files that target cljs
/// (requiring `cljs.core`, `cljs.test`, …) find our namespaces.
fn normalize_ns(name: &str) -> String {
    match name {
        "cljs.core" => "clojure.core".to_string(),
        "cljs.test" => "clojure.test".to_string(),
        _ => name.to_string(),
    }
}

/// Add refers for `[x y]` (a vector of names) under namespace `real`. When `kw`
/// is non-empty the list is preceded by that keyword marker (`:only [x y]`).
fn refer_names<M: ValueModel>(rt: &mut Runtime<M>, comp: &mut Compiler, forms: &[u64], real: &str, kw: &str) {
    let list = if kw.is_empty() {
        forms.first().copied()
    } else {
        forms.iter().position(|&f| is_keyword(rt, f, kw)).and_then(|i| forms.get(i + 1).copied())
    };
    let Some(list) = list else { return };
    if let Some(names) = binding_items(rt, unquote(rt, list)) {
        for nm in names {
            if let Some(short) = sym_name_of(rt, nm) {
                comp.add_refer(&short, &format!("{real}/{short}"));
            }
        }
    }
}

/// Load a required namespace from disk if it isn't already loaded. Maps ns
/// `foo.bar` to `foo/bar.clj` (`.cljc`/`.cljs` also tried; hyphens in a segment
/// become underscores, as Clojure munges file paths) and searches each load-path
/// directory. Errors clearly if the namespace can't be located.
fn ensure_loaded<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    name: &str,
) {
    if comp.is_loaded(name) {
        return;
    }
    let rel = name.replace('.', "/").replace('-', "_");
    // A load-path entry is a DIRECTORY of sources or a JAR (a Maven dep) —
    // exactly Clojure's classpath model.
    let mut src = None;
    for entry in comp.load_paths() {
        if entry.extension().is_some_and(|e| e == "jar") {
            if let Some(s) = deps::jar_source(entry, &rel) {
                src = Some(s);
            }
        } else {
            for ext in ["clj", "cljc", "cljs"] {
                let p = entry.join(format!("{rel}.{ext}"));
                if p.is_file() {
                    src = Some(std::fs::read_to_string(&p).unwrap_or_else(|e| {
                        panic!("require: failed reading {}: {e}", p.display())
                    }));
                    break;
                }
            }
        }
        if src.is_some() {
            break;
        }
    }
    let Some(src) = src else {
        panic!(
            "require: cannot find namespace `{name}` (looked for `{rel}.clj` on load path {:?})",
            comp.load_paths()
        );
    };
    // Mark loaded BEFORE running so a cyclic require terminates.
    comp.mark_loaded(name);
    let saved = comp.current_ns().to_string();
    run_src(rt, cs, macros, comp, &src);
    // The loaded file's `(ns …)` moved us into it; restore the requiring ns.
    comp.set_ns(&saved);
}

/// The fully-qualified class name for a class reference: dotted (or
/// `ns/Class`-qualified) names pass through; a bare simple name resolves via
/// the current ns's imports + the auto-import table, else stays itself
/// (deftype tags resolve at runtime by tag equality).
fn class_fqn<M: ValueModel>(rt: &Runtime<M>, comp: &Compiler, class_sym: Sym) -> (String, String) {
    let cname = rt.sym_name(class_sym).to_string();
    let simple = last_seg(&cname);
    let fqn = if cname.contains('.') || cname.contains('/') {
        cname.replace('/', ".")
    } else {
        comp.resolve_class(&simple).unwrap_or_else(|| simple.clone())
    };
    (fqn, simple)
}

/// `(instance? Class x)`. A class name resolving to a defined VAR (a protocol
/// or a deftype's type value) checks through `-instance-val`; anything else
/// asks the in-language JVM layer (`-jvm-instance-of?`: registered classes by
/// tag + inheritance walk, interfaces by protocol satisfaction, unknown names
/// by tag equality). A non-symbol class expression is a runtime class VALUE
/// (`Class` record / protocol / tag sym) — fully dynamic, like Clojure's fn.
fn instance_rewrite<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let Val::Sym(cs) = rt.decode(items[1]) else {
        let iv = sym(rt, "-jvm-instance?");
        return rt.vec_to_list(&[iv, items[1], items[2]]);
    };
    // If the class NAME resolves to a bound var, it may be a PROTOCOL (in which
    // case `instance?` means "satisfies the protocol") or a deftype's own type
    // value. `-instance-val` handles both at runtime; class names with no var
    // (host classes like `java.lang.String`) fall through to tag equality. A
    // Java-style dotted ref `a.b.C` may name the var `a.b/C` (e.g. a protocol).
    let cname = rt.sym_name(cs).to_string();
    let candidates = {
        let mut v = vec![comp.resolve_ref(rt, cs)];
        if let Some(pos) = cname.rfind('.') {
            let dotted_var = rt.intern(&format!("{}/{}", &cname[..pos], &cname[pos + 1..]));
            v.push(comp.resolve_ref(rt, dotted_var));
            // `clojure.lang.ILookup` etc.: the JVM interface corresponds to OUR
            // protocol of the same simple name (a clojure.core var), so
            // `instance?` means `satisfies?` — resolve the bare simple name too.
            let simple = rt.intern(&cname[pos + 1..]);
            v.push(comp.resolve_ref(rt, simple));
        }
        v
    };
    if let Some(&resolved) = candidates.iter().find(|&&r| rt.global_defined(r)) {
        let iv = sym(rt, "-instance-val");
        let cref = rt.encode(Val::Sym(resolved));
        return rt.vec_to_list(&[iv, cref, items[2]]);
    }
    let (fqn, simple) = class_fqn(rt, comp, cs);
    jvm_layer_call(rt, "-jvm-instance-of?", &[&fqn, &simple], &items[2..3])
}

/// Desugar `(try body… (catch Class e h…)… (catch :default e h…) (finally f…))`
/// into `(try* (do body…) EXC DISPATCH (do f…))` — a fixed-shape low-level form
/// the compiler maps to `Ir::Try`. DISPATCH is a nested-`if` that tests the
/// thrown value's runtime tag against each clause's class (ClojureScript's
/// `instanceof`-chain model), binds the clause's name, and re-`throw`s on no
/// match. A `:default` clause, or a base class (`Throwable`/`Exception`/`Error`/
/// `Object`), matches anything. `try*`/`EXC`/`DISPATCH`/`finally` are `nil` when
/// the corresponding part is absent.
fn desugar_try<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let mut body_forms: Vec<u64> = Vec::new();
    let mut catches: Vec<Vec<u64>> = Vec::new();
    let mut finally: Option<Vec<u64>> = None;
    for &f in &items[1..] {
        let head = rt.as_cons(f).map(|_| rt.list_to_vec(f)[0]);
        match head {
            Some(h) if is_sym(rt, h, "catch") => catches.push(rt.list_to_vec(f)),
            Some(h) if is_sym(rt, h, "finally") => finally = Some(rt.list_to_vec(f)[1..].to_vec()),
            _ => body_forms.push(f),
        }
    }
    let do_sym = sym(rt, "do");
    let nil = rt.encode(Val::Nil);
    let mut body = vec![do_sym];
    body.extend_from_slice(&body_forms);
    let body = rt.vec_to_list(&body);

    let (catchbind, dispatch) = if catches.is_empty() {
        (nil, nil)
    } else {
        let exc = gensym(rt, "exc");
        // Fold clauses back-to-front so the first listed clause is tried first.
        let throwsym = sym(rt, "throw");
        let mut chain = rt.vec_to_list(&[throwsym, exc]); // no clause matched -> re-throw
        for clause in catches.iter().rev() {
            // clause = (catch <class-or-:default> <bind> body…)
            let class = clause[1];
            let bind = clause[2];
            let test = catch_test(rt, comp, class, exc);
            // (let (bind exc) body…)
            let letsym = sym(rt, "let");
            let bindlist = rt.vec_to_list(&[bind, exc]);
            let mut letf = vec![letsym, bindlist];
            letf.extend_from_slice(&clause[3..]);
            let handler = rt.vec_to_list(&letf);
            let ifsym = sym(rt, "if");
            chain = rt.vec_to_list(&[ifsym, test, handler, chain]);
        }
        (exc, chain)
    };

    let finally_form = match finally {
        Some(fs) => {
            let mut f = vec![do_sym];
            f.extend_from_slice(&fs);
            rt.vec_to_list(&f)
        }
        None => nil,
    };

    let trystar = sym(rt, "try*");
    rt.vec_to_list(&[trystar, body, catchbind, dispatch, finally_form])
}

/// `(binding [*x* v …] body…)` -> `(try* (do (%dyn-mark) (%dyn-bind 'ns/*x* v) …
/// body…) nil nil (%dyn-unwind))`. The `%dyn-mark` delimits this scope on the
/// per-thread binding stack; the `finally` `%dyn-unwind` pops it (so bindings
/// unwind even when the body throws). Each var is RESOLVED to its qualified
/// dynamic sym so it keys the stack the same way references (`%dyn-get`) do.
fn desugar_binding<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let pairs = binding_items(rt, items[1]).unwrap_or_else(|| rt.list_to_vec(items[1]));
    let quote = sym(rt, "quote");
    let dynbind = sym(rt, "%dyn-bind");
    let do_sym = sym(rt, "do");
    let nil = rt.encode(Val::Nil);

    let mut body = vec![do_sym];
    let mark = sym(rt, "%dyn-mark");
    body.push(rt.vec_to_list(&[mark]));
    let mut k = 0;
    while k + 1 < pairs.len() {
        // Resolve the var to its qualified dynamic sym (matching `%dyn-get`).
        let resolved = match rt.decode(pairs[k]) {
            Val::Sym(s) => rt.encode(Val::Sym(comp.resolve_ref(rt, s))),
            _ => pairs[k],
        };
        let qvar = rt.vec_to_list(&[quote, resolved]);
        body.push(rt.vec_to_list(&[dynbind, qvar, pairs[k + 1]]));
        k += 2;
    }
    body.extend_from_slice(&items[2..]);
    let body = rt.vec_to_list(&body);

    let unwind = {
        let u = sym(rt, "%dyn-unwind");
        rt.vec_to_list(&[u])
    };
    let trystar = sym(rt, "try*");
    rt.vec_to_list(&[trystar, body, nil, nil, unwind])
}

/// `(with-redefs [f newf …] body…)` -> save each var's current root, install the
/// replacements, run the body, and (in a `finally`) restore the roots. Operates on
/// var ROOTS by symbol (`(var f)` resolves the name), so it composes with the
/// namespace-qualified var model.
fn desugar_with_redefs<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let pairs = binding_items(rt, items[1]).unwrap_or_else(|| rt.list_to_vec(items[1]));
    let var_s = sym(rt, "var");
    let vget = sym(rt, "var-get");
    let vset = sym(rt, "var-set");
    let let_s = sym(rt, "let");
    let do_s = sym(rt, "do");
    let trystar = sym(rt, "try*");
    let nil = rt.encode(Val::Nil);

    let mut let_binds = Vec::new(); // [old_i (var-get (var name_i)) …]
    let mut sets = Vec::new(); // (var-set (var name_i) new_i)
    let mut restores = Vec::new(); // (var-set (var name_i) old_i)
    let mut k = 0;
    while k + 1 < pairs.len() {
        let (name, new) = (pairs[k], pairs[k + 1]);
        let old = gensym(rt, "old");
        let vf = |rt: &mut Runtime<M>| rt.vec_to_list(&[var_s, name]);
        let getf = {
            let v = vf(rt);
            rt.vec_to_list(&[vget, v])
        };
        let_binds.push(old);
        let_binds.push(getf);
        let setf = {
            let v = vf(rt);
            rt.vec_to_list(&[vset, v, new])
        };
        sets.push(setf);
        let restf = {
            let v = vf(rt);
            rt.vec_to_list(&[vset, v, old])
        };
        restores.push(restf);
        k += 2;
    }
    let finally = {
        let mut fin = vec![do_s];
        fin.extend(restores);
        rt.vec_to_list(&fin)
    };
    let body = {
        let mut b = vec![do_s];
        b.extend_from_slice(&items[2..]);
        rt.vec_to_list(&b)
    };
    let tryf = rt.vec_to_list(&[trystar, body, nil, nil, finally]);
    let bindvec = make_vector(rt, let_binds);
    let mut letform = vec![let_s, bindvec];
    letform.extend(sets);
    letform.push(tryf);
    rt.vec_to_list(&letform)
}

/// The test form for one catch clause. `:default` and the throwable ROOTS
/// (Throwable/Exception/Error/Object) are compile-time `true` — thrown values
/// in this dialect include strings and plain records, so the roots must catch
/// EVERYTHING (dialect semantics, not a registry question). A specific class
/// asks the JVM layer: tag match, or superclass/interface walk through the
/// registry, or plain tag equality for unregistered (deftype) names.
fn catch_test<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, class: u64, exc: u64) -> u64 {
    let catch_all = if is_keyword(rt, class, "default") {
        true
    } else if let Val::Sym(s) = rt.decode(class) {
        let simple = last_seg(rt.sym_name(s));
        matches!(simple.as_str(), "Throwable" | "Exception" | "Error" | "Object")
    } else {
        false
    };
    if catch_all {
        return rt.encode(Val::Bool(true));
    }
    let Val::Sym(s) = rt.decode(class) else {
        panic!("catch: expected a class symbol or :default");
    };
    let (fqn, simple) = class_fqn(rt, comp, s);
    jvm_layer_call(rt, "-jvm-catch-match?", &[&fqn, &simple], &[exc])
}

/// `(def (-macro-meta name) val…)` -> `(name, (def name val…))`, else None.
fn strip_def_macro_meta<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> Option<(Sym, u64)> {
    let items = rt.list_to_vec(form);
    if items.len() < 2 || !is_sym(rt, items[0], "def") {
        return None;
    }
    let nparts = rt.list_to_vec(items[1]);
    if nparts.len() < 2 || !is_sym(rt, nparts[0], "-macro-meta") {
        return None;
    }
    let name = match rt.decode(nparts[1]) {
        Val::Sym(s) => s,
        _ => return None,
    };
    let mut newf = vec![items[0], nparts[1]];
    newf.extend_from_slice(&items[2..]);
    Some((name, rt.vec_to_list(&newf)))
}

/// Recognize real core.clj's macro registration `(. (var NAME) (setMacro))`
/// (and its `(. (var NAME) setMacro)` / `(.setMacro (var NAME))` variants),
/// returning NAME. This runs at eval-form level because marking a macro must
/// mutate the (frontend-owned) macro set that the NEXT form's expander reads.
fn setmacro_target<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Sym> {
    let items = rt.list_to_vec(form);
    if items.is_empty() {
        return None;
    }
    // Locate the receiver form and confirm the method is `setMacro`.
    let recv = if is_sym(rt, items[0], ".") {
        if items.len() < 3 {
            return None;
        }
        // items[2] is either `(setMacro)` or the bare symbol `setMacro`.
        let is_setmacro = match rt.as_cons(items[2]) {
            Some(_) => is_sym(rt, rt.list_to_vec(items[2])[0], "setMacro"),
            None => is_sym(rt, items[2], "setMacro"),
        };
        if !is_setmacro {
            return None;
        }
        items[1]
    } else if is_sym(rt, items[0], ".setMacro") {
        if items.len() < 2 {
            return None;
        }
        items[1]
    } else {
        return None;
    };
    // The receiver must be `(var NAME)` or a `#'NAME` var literal.
    let rv = rt.list_to_vec(recv);
    if rv.len() == 2 && is_sym(rt, rv[0], "var") {
        if let Val::Sym(name) = rt.decode(rv[1]) {
            return Some(name);
        }
    }
    None
}

fn defmacro_name<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Sym> {
    let (h, _) = rt.as_cons(form)?;
    if !is_sym(rt, h, "defmacro") {
        return None;
    }
    let items = rt.list_to_vec(form);
    match rt.decode(items[1]) {
        Val::Sym(name) => Some(name),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Clojure-flavored printing (the toolkit's `print` doesn't know these tags).
// ─────────────────────────────────────────────────────────────────────────

/// Render a value the way Clojure would: `[..]` vectors, `{..}` maps, `#{..}`
/// sets, `:kw` keywords, `(..)` lists.
pub fn clj_str<M: ValueModel>(rt: &Runtime<M>, v: u64) -> String {
    match rt.decode(v) {
        Val::Ref(id) => match rt.view_gc(id) {
            ObjView::Record { type_id, fields } => {
                let tag = rt.sym_name(type_id);
                match tag {
                    "Keyword" => format!(":{}", rt.print(fields[0])),
                    "Vector" => format!("[{}]", list_items(rt, fields[0], " ")),
                    "Set" => format!("#{{{}}}", list_items(rt, fields[0], " ")),
                    "Map" => format!("{{{}}}", map_items(rt, fields[0])),
                    "SortedSet" => format!("#{{{}}}", list_items(rt, fields[0], " ")),
                    "SortedMap" => format!("{{{}}}", map_items(rt, fields[0])),
                    "Var" => format!("#'{}", rt.print(fields[0])),
                    _ => rt.print(v),
                }
            }
            ObjView::Cons { .. } => format!("({})", list_items(rt, v, " ")),
            // A char prints readably as `\a` (like Clojure's pr), distinct from
            // `(str \a)` -> "a".
            ObjView::Char(c) => match c {
                ' ' => "\\space".to_string(),
                '\n' => "\\newline".to_string(),
                '\t' => "\\tab".to_string(),
                '\r' => "\\return".to_string(),
                _ => format!("\\{c}"),
            },
            _ => rt.print(v),
        },
        _ => rt.print(v),
    }
}

fn list_items<M: ValueModel>(rt: &Runtime<M>, lst: u64, sep: &str) -> String {
    rt.list_to_vec(lst)
        .iter()
        .map(|&x| clj_str(rt, x))
        .collect::<Vec<_>>()
        .join(sep)
}

fn map_items<M: ValueModel>(rt: &Runtime<M>, kvs: u64) -> String {
    let items = rt.list_to_vec(kvs);
    items
        .chunks(2)
        .map(|kv| {
            let k = clj_str(rt, kv[0]);
            let v = kv.get(1).map(|&x| clj_str(rt, x)).unwrap_or_default();
            format!("{k} {v}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}
