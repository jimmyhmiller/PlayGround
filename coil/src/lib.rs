//! Coil — a low-level Lisp where calling convention (and, later, allocation) is
//! part of the type system. This crate is the M0/M1 skeleton: reader → core →
//! convention-aware checks → LLVM codegen that JIT-runs `main`.
//!
//! See `docs/DESIGN.md` for the full design.

pub mod abi;
pub mod ast;
pub mod check;
pub mod cheader;
pub mod cimport;
pub mod codegen;
pub mod comptime;
pub mod convention;
pub mod dump_ast;
pub mod dump_expand;
pub mod dump_load;
pub mod dump_resolved;
pub mod dump_checked;
pub mod dump_mono;
pub mod macros;
pub mod manifest;
pub mod mono;
pub mod normalize_ir;
pub mod parse;
pub mod reader;
pub mod repl;
pub mod resolve;
pub mod span;
pub mod stdlib;

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;

/// The LLVM new-pass-manager pipeline Coil runs over every module before it
/// emits an object. `emit_ir` deliberately skips this (it shows the raw,
/// readable IR we generate, which the struct-ABI tests diff against clang's
/// unoptimized output); the *compiled* output is fully optimized.
const OPT_PIPELINE: &str = "default<O3>";
use std::path::Path;
use std::process::Command;

use span::{Diag, SourceMap};

/// Render any pipeline error against the `SourceMap` into a finished diagnostic
/// string (`file:line:col` + caret per error, plus an `N errors` summary when there
/// is more than one). The main file is registered under the `<source>` placeholder,
/// which the CLI substitutes for the real path; imported files carry their real path.
fn reported<T>(r: Result<T, Vec<Diag>>, sm: &SourceMap) -> Result<T, String> {
    r.map_err(|diags| span::render_all(&diags, sm))
}

/// Lift a single-error pass result into the multi-error spine.
fn one<T>(r: Result<T, Diag>) -> Result<T, Vec<Diag>> {
    r.map_err(|d| vec![d])
}

/// Collapse a multi-error checker result to one diagnostic — for the internal
/// sub-program checks (macro/comptime contexts) that are reported single-error.
fn collapse(diags: Vec<Diag>) -> Diag {
    diags.into_iter().next().unwrap_or_else(|| Diag::new("type error"))
}

/// Time a build phase, printing `[time] <label>: <ms>` to stderr when `COIL_TIME` is
/// set in the environment — a cheap always-available profiler that splits the front
/// end (read→expand→check→mono) from codegen, the LLVM `-O3` pipeline, object
/// emission, and linking, to see where `coil build` spends its time.
fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if std::env::var_os("COIL_TIME").is_none() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    eprintln!("  [time] {label}: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    r
}

/// The whole front end: read → macro-expand → resolve → check, with the Stage-3
/// elaboration loop for `(meta …)` staged macros. Returns the checked `Program`.
///
/// Staged macros need generated definitions in place *before* the code that
/// depends on them is checked. So when there are `(meta …)` forms: check the
/// closure of functions the metas reach (the generators — they don't depend on
/// generated code, so they check cleanly), run the metas via the comptime
/// interpreter to get generated top-level forms, splice those in (dropping the
/// `meta` forms), and check the whole program.
fn elaborate(src: &str, sm: &mut SourceMap) -> Result<ast::Program, Vec<Diag>> {
    // The main file is source 0, registered under the `<source>` placeholder (the CLI
    // swaps in the real path; tests render against `<source>`). Imports + the prelude
    // register themselves as further sources inside `load_program`.
    let main = sm.add("<source>", src);
    // Run the front end on a large dedicated stack. Type-checking recurses over
    // expression nesting, and the auto-loaded prelude (`coil.core`) now always pulls
    // in the comptime macro engine — `fmt`'s deeply-nested format walker and friends —
    // so a default 2 MiB host thread (cargo's test threads) can otherwise overflow on
    // even a trivial program. (The macro expander already does this for the same
    // reason.) A real program with deep nesting is likewise protected.
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn_scoped(s, || elaborate_on_stack(src, main, sm))
            .expect("spawn front-end thread")
            .join()
            .expect("front-end thread panicked")
    })
}

/// The body of `elaborate`, run on a large stack (see `elaborate`).
fn elaborate_on_stack(src: &str, main: u32, sm: &mut SourceMap) -> Result<ast::Program, Vec<Diag>> {
    let (program, imports, exports) = front_end_on_stack(src, main, sm)?;
    let mut checked = check::check_with(&program, &imports, &exports)?;
    // Comptime-only functions (anything with a `Code` parameter or return — macros,
    // generators, code helpers) have done their job during elaboration and have no
    // runtime representation; drop them before mono/codegen.
    checked.funcs.retain(|f| f.ret != ast::Type::Code && f.params.iter().all(|p| p.ty != ast::Type::Code));
    Ok(checked)
}

/// The front end WITHOUT the final whole-program type-check: read → load →
/// stage-3 expand → resolve, including the staged-macro elaboration loop (which
/// internally checks the *generator* closure before running the metas). Returns
/// the final resolved program plus the import/export tables the check needs.
/// `elaborate_on_stack` = this + `check_with`; the REPL's type-inference probe
/// (`repl_infer_tail`) runs `check::infer_tail_type` on the result instead — its
/// probe function deliberately has no meaningful declared return type, so the
/// whole-program check (which enforces declared returns) cannot run over it.
fn front_end_on_stack(
    src: &str,
    main: u32,
    sm: &mut SourceMap,
) -> Result<(ast::Program, macros::ImportMap, macros::ExportMap), Vec<Diag>> {
    // The front-end pipeline up to the checker is fail-fast (single `Diag`); the
    // checker (`check_with`) returns *all* of one program's body errors, which flow
    // straight out as the multi-error `Vec<Diag>`.
    let forms = one(reader::read_all(src, main))?;
    let (tagged, imports, exports) = one(macros::load_program(&forms, &host_target(), sm))?;
    let tagged = one(expand_stage3_macros(tagged, &imports, &exports, sm))?;
    // With `(meta …)` generators present, this first resolve is intermediate — the
    // program still calls definitions the metas haven't generated yet — so the
    // undefined-reference check is deferred to the final `program2` resolve below.
    let has_metas = tagged.iter().any(|(f, _)| is_meta_form(f));
    let program = one(resolve::resolve_program(tagged.clone(), &imports, &exports, !has_metas))?;

    // ---- top-level `(meta …)` staged macros (generate definitions) ----
    if program.metas.is_empty() {
        return Ok((program, imports, exports));
    }
    let mut stack: Vec<String> = Vec::new();
    let names: std::collections::HashSet<&str> =
        program.funcs.iter().map(|f| f.name.as_str()).collect();
    for m in &program.metas {
        collect_calls(m, &names, &mut stack);
    }
    let wanted = closure_funcs(&program, stack);
    let sub = closure_subprogram(&program, &wanted, true);
    let checked_sub = check::check(&sub)?;
    let gen = one(comptime::run_metas(&checked_sub).map_err(crate::span::Diag::new))?;
    let module = tagged.iter().find(|(f, _)| is_meta_form(f)).and_then(|(_, m)| m.clone());
    let mut tagged2: Vec<macros::TaggedForm> =
        tagged.into_iter().filter(|(f, _)| !is_meta_form(f)).collect();
    for g in gen {
        tagged2.push((g, module.clone()));
    }
    let program2 = one(resolve::resolve_program(tagged2, &imports, &exports, true))?;
    Ok((program2, imports, exports))
}

/// Keep a form for macro detection iff it parses without depending on macros being
/// expanded first: pure macro/helper defns (`(defn … (-> Code) …)`) and plain
/// declarations. Runtime defns, impls, consts, metas have expression bodies that
/// may contain not-yet-expanded macro calls (with binding-vector / keyword args).
fn keep_for_macro_detection(f: &reader::Sexp) -> bool {
    use reader::SexpKind as K;
    let items = match &f.kind {
        K::List(it) => it,
        _ => return true,
    };
    let head = match items.first().map(|x| &x.kind) {
        Some(K::Sym(h)) => h.as_str(),
        _ => return true,
    };
    match head {
        "defn" => defn_involves_code(items),
        // declarations the macros may reference — always parseable, no macro calls
        "defstruct" | "defsum" | "deftrait" | "extern" | "module" | "import" | "include"
        | "export" | "export-c" | "defcc" => true,
        // impls/consts/metas have runtime bodies; a bare `(macro …)` call is not a
        // definition — both may contain not-yet-expanded macro calls, so drop them.
        _ => false,
    }
}

/// A `(defn …)` whose signature mentions `Code` — a macro or comptime helper (the
/// only things that take/return `Code`). Its body parses without prior expansion.
fn defn_involves_code(items: &[reader::Sexp]) -> bool {
    use reader::SexpKind as K;
    let is_code = |s: &reader::Sexp| matches!(&s.kind, K::Sym(t) if t == "Code");
    items.iter().any(|it| match &it.kind {
        // return type `(-> Code)`
        K::List(l) => {
            matches!(l.first().map(|x| &x.kind), Some(K::Sym(h)) if h == "->")
                && l.get(1).is_some_and(is_code)
        }
        // a param vector with a `(name Code)` entry
        K::Vector(ps) => ps.iter().any(|p| match &p.kind {
            K::List(pair) => pair.get(1).is_some_and(is_code),
            _ => false,
        }),
        _ => false,
    })
}

/// Expand expression-position Stage-3 macros over the raw (post-old-expand) forms.
/// A macro is a `[Code…] -> Code` function; its calls expand inline. We must expand
/// BEFORE parsing the full program (a macro call with non-expression args — a
/// `[i lo hi]` vector, a `:keyword` — doesn't parse until expanded), so detect +
/// check the macros from a PARSEABLE subset (declarations + comptime defns), then
/// expand every form. Returns the macro-free forms.
fn expand_stage3_macros(
    tagged: Vec<macros::TaggedForm>,
    imports: &macros::ImportMap,
    exports: &macros::ExportMap,
    sm: &mut SourceMap,
) -> Result<Vec<macros::TaggedForm>, Diag> {
    // The comptime interpreter recurses (Rust stack) for each step of a macro's
    // evaluation — a macro that loops over, say, a format string's characters can
    // go deep. Run on a large dedicated stack so it's robust off the main thread
    // (e.g. inside cargo's 2 MiB test threads). The `SourceMap` is borrowed into the
    // scoped thread so macro expansions can register provenance records.
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn_scoped(s, move || expand_stage3_macros_inner(tagged, imports, exports, sm))
            .expect("spawn macro-expansion thread")
            .join()
            .expect("macro-expansion thread panicked")
    })
}

fn expand_stage3_macros_inner(
    tagged: Vec<macros::TaggedForm>,
    imports: &macros::ImportMap,
    exports: &macros::ExportMap,
    sm: &mut SourceMap,
) -> Result<Vec<macros::TaggedForm>, Diag> {
    use std::collections::HashSet;
    let macro_subset: Vec<macros::TaggedForm> =
        tagged.iter().filter(|(f, _)| keep_for_macro_detection(f)).cloned().collect();
    let macro_ctx = resolve::resolve_program(macro_subset, imports, exports, false)?;
    let macro_quals: Vec<String> = macro_ctx
        .funcs
        .iter()
        .filter(|f| f.ret == ast::Type::Code && f.params.iter().all(|p| p.ty == ast::Type::Code))
        .map(|f| f.name.clone())
        .collect();
    if macro_quals.is_empty() {
        return Ok(tagged);
    }
    // The qualified names of all macros, and the comptime closure (macros + the
    // helpers they call) whose bodies are left unexpanded (run at expansion time).
    let macro_set: HashSet<String> = macro_quals.iter().cloned().collect();
    let wanted = closure_funcs(&macro_ctx, macro_quals);
    let sub = closure_subprogram(&macro_ctx, &wanted, false);
    let checked_sub = check::check(&sub).map_err(collapse)?;
    // Resolving a macro call respects module scope, exactly like the name resolver:
    // a bare `(name …)` is a macro call only if `name` is a macro visible in the
    // call's module (own module, or a `:use`d module's export); `alias/name` goes
    // through `:as`; a `.`-qualified head comes from hygiene.
    let env = MacroEnv { macros: &macro_set, comptime: &wanted, imports, exports, sub: &checked_sub };
    // All qualified function names (for referential hygiene of generated calls).
    // GROWS as macros generate defns, so hygiene also covers macro-generated fns.
    let qualify = |n: String, m: &Option<String>| match m {
        Some(md) => format!("{md}.{n}"),
        None => n,
    };
    let mut all_fns: HashSet<String> =
        tagged.iter().filter_map(|(f, m)| defn_name(f).map(|n| qualify(n, m))).collect();
    let mut expanded: Vec<macros::TaggedForm> = Vec::with_capacity(tagged.len());
    // Declarations generated by earlier macros, so a later macro can reflect on a
    // macro-generated struct/sum (incremental, like the old def table).
    let mut gen: (Vec<ast::StructDef>, Vec<ast::SumDef>) = (Vec::new(), Vec::new());
    for (f, m) in tagged {
        let out = expand_top_form(&f, m.as_deref(), &env, &all_fns, &gen, &mut 0, sm)?;
        // A macro that expands to a top-level `(do …)` splices into several forms.
        // Record each result's generated decls + fn names so LATER forms see them.
        let mut produced: Vec<reader::Sexp> = Vec::new();
        match &out.kind {
            reader::SexpKind::List(items)
                if matches!(items.first().map(|x| &x.kind), Some(reader::SexpKind::Sym(h)) if h == "do") =>
            {
                produced.extend(items[1..].iter().cloned());
            }
            _ => produced.push(out),
        }
        for form in produced {
            register_generated_decl(&form, &m, &mut gen)?;
            if let Some(n) = defn_name(&form) {
                all_fns.insert(qualify(n, &m));
            }
            expanded.push((form, m.clone()));
        }
    }
    Ok(expanded)
}

/// If `form` is a `(defstruct …)`/`(defsum …)`, parse it and record it (with its
/// name QUALIFIED by `module`, matching how the resolver keys types) so later macros
/// in the same pass can reflect on it by the same qualified key. A parse failure
/// here is a real error in the generating macro's output — surface it, don't swallow.
fn register_generated_decl(
    form: &reader::Sexp,
    module: &Option<String>,
    gen: &mut (Vec<ast::StructDef>, Vec<ast::SumDef>),
) -> Result<(), Diag> {
    let head = match &form.kind {
        reader::SexpKind::List(items) => items.first().and_then(|x| match &x.kind {
            reader::SexpKind::Sym(h) => Some(h.as_str()),
            _ => None,
        }),
        _ => None,
    };
    if !matches!(head, Some("defstruct") | Some("defsum")) {
        return Ok(());
    }
    let qualify = |n: &str| match module {
        Some(m) => format!("{m}.{n}"),
        None => n.to_string(),
    };
    let p = parse::parse_program(std::slice::from_ref(form))?;
    for mut s in p.structs {
        s.name = qualify(&s.name);
        gen.0.push(s);
    }
    for mut s in p.sums {
        s.name = qualify(&s.name);
        gen.1.push(s);
    }
    Ok(())
}

fn is_meta_form(s: &reader::Sexp) -> bool {
    matches!(&s.kind, reader::SexpKind::List(items)
        if matches!(items.first().map(|x| &x.kind), Some(reader::SexpKind::Sym(h)) if h == "meta"))
}

/// The closure of function names reachable (via calls) from the seed names.
fn closure_funcs(p: &ast::Program, mut stack: Vec<String>) -> std::collections::HashSet<String> {
    let names: std::collections::HashSet<&str> = p.funcs.iter().map(|f| f.name.as_str()).collect();
    let by_name: std::collections::HashMap<&str, &ast::Func> =
        p.funcs.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut wanted = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if wanted.insert(n.clone()) {
            if let Some(f) = by_name.get(n.as_str()) {
                for e in &f.body {
                    collect_calls(e, &names, &mut stack);
                }
            }
        }
    }
    wanted
}

/// A sub-program with only the `wanted` functions (plus full type/trait/const
/// context), so a comptime stage can be checked without the code that consumes it.
fn closure_subprogram(
    p: &ast::Program,
    wanted: &std::collections::HashSet<String>,
    keep_metas: bool,
) -> ast::Program {
    ast::Program {
        conventions: p.conventions.clone(),
        structs: p.structs.clone(),
        sums: p.sums.clone(),
        externs: p.externs.clone(),
        funcs: p.funcs.iter().filter(|f| wanted.contains(&f.name)).cloned().collect(),
        asserts: vec![],
        consts: p.consts.clone(),
        traits: p.traits.clone(),
        impls: p.impls.clone(),
        statics: vec![],
        metas: if keep_metas { p.metas.clone() } else { vec![] },
        exports: vec![], // a comptime sub-program has no C exports
    }
}

/// The immutable context for macro expansion: the qualified macro names, the
/// comptime closure (macros + their helpers), the import/export tables (for scoped
/// resolution), and the checked sub-program the macros run in.
struct MacroEnv<'a> {
    macros: &'a std::collections::HashSet<String>,
    comptime: &'a std::collections::HashSet<String>,
    imports: &'a macros::ImportMap,
    exports: &'a macros::ExportMap,
    sub: &'a ast::Program,
}

/// Resolve a call head to the qualified macro it names, scoped to `module` exactly
/// like the name resolver — or `None` if it isn't a macro visible there.
fn resolve_macro(head: &str, module: Option<&str>, env: &MacroEnv) -> Option<String> {
    // `alias/name` → through this module's `:as` aliases, export-checked.
    if let Some((alias, rest)) = head.split_once('/') {
        let target = env.imports.get(module?)?.aliases.get(alias)?;
        let q = format!("{target}.{rest}");
        return (env.macros.contains(&q) && macros::exports(env.exports, target, rest)).then_some(q);
    }
    // A `.`-qualified head is hygiene-generated (already a full `module.name`).
    if head.contains('.') {
        return env.macros.contains(head).then(|| head.to_string());
    }
    // Bare: own module first (module-less programs name macros bare).
    let own = match module {
        Some(m) => format!("{m}.{head}"),
        None => head.to_string(),
    };
    if env.macros.contains(&own) {
        return Some(own);
    }
    // …then this module's `:use`d exported macros.
    if let Some(imp) = env.imports.get(module.unwrap_or("")) {
        for (target, spec) in &imp.uses {
            let used = match spec {
                macros::UseSpec::All => true,
                macros::UseSpec::Names(ns) => ns.iter().any(|n| n == head),
            };
            let q = format!("{target}.{head}");
            if used && macros::exports(env.exports, target, head) && env.macros.contains(&q) {
                return Some(q);
            }
        }
    }
    // …finally, macros from modules `coil.core` explicitly `:reexport`s — these are
    // auto-referred everywhere (that's the one line in the prelude that makes
    // control flow free). Only `:reexport`ed targets count, NOT coil.core's `:use`s.
    if module != Some("coil.core") {
        if let Some(core) = env.imports.get("coil.core") {
            for target in &core.reexports {
                let q = format!("{target}.{head}");
                if macros::exports(env.exports, target, head) && env.macros.contains(&q) {
                    return Some(q);
                }
            }
        }
    }
    None
}

/// Expand expression-macro calls in one top-level form. Macro definitions and
/// `(meta …)` forms are left alone (their bodies are comptime code, not user code).
fn expand_top_form(
    f: &reader::Sexp,
    module: Option<&str>,
    env: &MacroEnv,
    all_fns: &std::collections::HashSet<String>,
    gen: &(Vec<ast::StructDef>, Vec<ast::SumDef>),
    depth: &mut u32,
    sm: &mut SourceMap,
) -> Result<reader::Sexp, Diag> {
    if is_meta_form(f) || is_comptime_defn(f, module, env.comptime) {
        return Ok(f.clone());
    }
    expand_calls(f, module, env, all_fns, gen, depth, sm)
}

/// Tag every macro-synthesized node (a template list/vector wrapper, which the
/// quasiquote builder leaves `DUMMY`) with the expansion `ctxt` and locate it at the
/// macro `call` site. Nodes that already carry a span are left untouched: a template
/// *literal* keeps its real location in the macro's defining file (Stage A), and —
/// crucially — code spliced in via `~unquote` keeps the *caller's* span, so a type
/// error in a user expression passed to a macro is never mis-attributed to the macro.
fn stamp_expansion(s: &mut reader::Sexp, call: span::Span, ctxt: u32) {
    if s.span.is_dummy() {
        s.span = span::Span { source: call.source, lo: call.lo, hi: call.hi, ctxt };
    }
    match &mut s.kind {
        reader::SexpKind::List(items) | reader::SexpKind::Vector(items) => {
            for it in items {
                stamp_expansion(it, call, ctxt);
            }
        }
        _ => {}
    }
}

/// The defined name of a `(defn NAME …)` form, else `None`.
fn defn_name(f: &reader::Sexp) -> Option<String> {
    if let reader::SexpKind::List(items) = &f.kind {
        if let (Some(reader::SexpKind::Sym(h)), Some(reader::SexpKind::Sym(n))) =
            (items.first().map(|x| &x.kind), items.get(1).map(|x| &x.kind))
        {
            if h == "defn" {
                return Some(n.clone());
            }
        }
    }
    None
}

/// A `(defn NAME …)` in `module` whose qualified name is in the comptime closure (a
/// macro or a helper a macro calls). Its body is comptime code — leave its calls
/// unexpanded; they run when the macro itself runs.
fn is_comptime_defn(
    f: &reader::Sexp,
    module: Option<&str>,
    comptime: &std::collections::HashSet<String>,
) -> bool {
    if let reader::SexpKind::List(items) = &f.kind {
        if let (Some(reader::SexpKind::Sym(h)), Some(reader::SexpKind::Sym(n))) =
            (items.first().map(|x| &x.kind), items.get(1).map(|x| &x.kind))
        {
            let qualified = match module {
                Some(m) => format!("{m}.{n}"),
                None => n.clone(),
            };
            return h == "defn" && comptime.contains(&qualified);
        }
    }
    false
}

/// Recursively expand `(macro args…)` calls outside-in: run the macro on its RAW
/// (unexpanded) argument forms — so a macro like `scope` can inspect them as data —
/// then re-expand the macro's output (which is where nested macro calls expand).
fn expand_calls(
    s: &reader::Sexp,
    module: Option<&str>,
    env: &MacroEnv,
    all_fns: &std::collections::HashSet<String>,
    gen: &(Vec<ast::StructDef>, Vec<ast::SumDef>),
    depth: &mut u32,
    sm: &mut SourceMap,
) -> Result<reader::Sexp, Diag> {
    use reader::{Sexp, SexpKind};
    match &s.kind {
        SexpKind::List(items) => {
            let mac = match items.first().map(|x| &x.kind) {
                Some(SexpKind::Sym(h)) => resolve_macro(h, module, env),
                _ => None,
            };
            if let Some(q) = mac {
                *depth += 1;
                if *depth > 100_000 {
                    return Err(Diag::new("macro expansion did not terminate"));
                }
                // RAW args (outside-in): the macro decides what to expand.
                let args: Vec<Sexp> = items[1..].to_vec();
                let mut out = comptime::expand_macro(env.sub, &q, args, all_fns, &gen.0, &gen.1, env.imports, env.exports)
                    .map_err(Diag::new)?;
                // Provenance: record this expansion (macro name + call site, chained to
                // the call site's own context for nested macros) and tag the
                // synthesized output with it, so a diagnostic landing in generated code
                // traces back here. `s.span` is the macro call's location.
                // The macro's definition span (its `(defn …)` template), so the trace
                // can point at the code that produced the offending node.
                let def_site = env.sub.funcs.iter()
                    .find(|f| f.name == q)
                    .map(|f| f.span)
                    .unwrap_or(span::Span::DUMMY);
                let ctxt = sm.add_expansion(span::Expansion {
                    macro_name: q.clone(),
                    call_site: s.span,
                    def_site,
                    parent: s.span.ctxt,
                });
                stamp_expansion(&mut out, s.span, ctxt);
                expand_calls(&out, module, env, all_fns, gen, depth, sm)
            } else {
                let out: Vec<Sexp> = items
                    .iter()
                    .map(|i| expand_calls(i, module, env, all_fns, gen, depth, sm))
                    .collect::<Result<_, _>>()?;
                Ok(Sexp::new(SexpKind::List(out), s.span))
            }
        }
        SexpKind::Vector(items) => {
            let out: Vec<Sexp> = items
                .iter()
                .map(|i| expand_calls(i, module, env, all_fns, gen, depth, sm))
                .collect::<Result<_, _>>()?;
            Ok(Sexp::new(SexpKind::Vector(out), s.span))
        }
        _ => Ok(s.clone()),
    }
}

/// Collect the names of called functions (restricted to `names`) in `e`.
fn collect_calls(e: &ast::Expr, names: &std::collections::HashSet<&str>, out: &mut Vec<String>) {
    use ast::ExprKind as K;
    let go = |x: &ast::Expr, out: &mut Vec<String>| collect_calls(x, names, out);
    match &e.kind {
        K::Call { func, args, .. } => {
            if names.contains(func.as_str()) {
                out.push(func.clone());
            }
            for a in args {
                go(a, out);
            }
        }
        K::Borrow { place, .. } => go(place, out),
        K::SpillRef(x) | K::Not(x) | K::Load(x) | K::Free(x) | K::Comptime(x) => go(x, out),
        K::Erase { inner, .. } | K::MakeDyn { inner, .. } => go(inner, out),
        K::DynDispatch { recv, args, .. } => {
            go(recv, out);
            for a in args {
                go(a, out);
            }
        }
        K::Cast { expr, .. } => go(expr, out),
        K::Bin { lhs, rhs, .. } | K::Cmp { lhs, rhs, .. } => {
            go(lhs, out);
            go(rhs, out);
        }
        K::If { cond, then, els } => {
            go(cond, out);
            go(then, out);
            go(els, out);
        }
        K::Do(es) | K::Loop { body: es, .. } => {
            for x in es {
                go(x, out);
            }
        }
        K::Break { value: Some(v), .. } => go(v, out),
        K::Let { binds, body } => {
            for (_, _, v) in binds {
                go(v, out);
            }
            for x in body {
                go(x, out);
            }
        }
        K::Field { ptr, .. } | K::BitGet { ptr, .. } => go(ptr, out),
        K::Store { ptr, val } => {
            go(ptr, out);
            go(val, out);
        }
        K::Index { ptr, idx } => {
            go(ptr, out);
            go(idx, out);
        }
        K::BitSet { ptr, val, .. } => {
            go(ptr, out);
            go(val, out);
        }
        K::Construct { args, .. } | K::CodeOp { args, .. } | K::TraitCall { args, .. } | K::LlvmIr { args, .. } => {
            for a in args {
                go(a, out);
            }
        }
        K::CallPtr { fp, args } => {
            go(fp, out);
            for a in args {
                go(a, out);
            }
        }
        K::Match { scrut, arms } => {
            go(scrut, out);
            for arm in arms {
                go(&arm.body, out);
            }
        }
        K::FieldMeta { idx, .. } => go(idx, out),
        K::FieldIndex { name, .. } => go(name, out),
        K::Quasi(q) => collect_calls_quasi(q, names, out),
        // leaves / no function calls
        K::Int(_) | K::Str(_) | K::CStr(_) | K::Var(_) | K::Float(_) | K::Bool(_) | K::Zeroed(_)
        | K::Alloc { .. } | K::SizeOf(_) | K::AlignOf(_) | K::OffsetOf(_, _) | K::FnPtrOf(_)
        | K::Continue { .. } | K::Break { value: None, .. } | K::StaticRef(_)
        | K::Quote(_) | K::TypeQuery { .. } => {}
    }
}

fn collect_calls_quasi(q: &ast::Quasi, names: &std::collections::HashSet<&str>, out: &mut Vec<String>) {
    match q {
        ast::Quasi::Lit(_) => {}
        ast::Quasi::Unquote(e) | ast::Quasi::Splice(e) => collect_calls(e, names, out),
        ast::Quasi::List(items) | ast::Quasi::Vector(items) => {
            for it in items {
                collect_calls_quasi(it, names, out);
            }
        }
    }
}

/// The compile-time target description handed to the macro evaluator. Derived
/// from the host triple (Coil currently AOT-targets the host).
fn host_target() -> macros::TargetInfo {
    let triple = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .into_owned();
    let mut parts = triple.split('-');
    let arch = parts.next().unwrap_or("unknown").to_string();
    let _vendor = parts.next();
    let os = parts.next().unwrap_or("unknown").to_string();
    let pointer_width = match arch.as_str() {
        "i386" | "i686" | "arm" | "armv7" | "thumbv7" | "wasm32" | "riscv32" | "mips" | "mipsel" => {
            32
        }
        _ => 64,
    };
    macros::TargetInfo {
        arch,
        os,
        triple,
        pointer_width,
    }
}

/// The shared front end: read → expand → parse → check (elaborate + infer) →
/// monomorphize → module. The checker runs *before* monomorphization now: it
/// types polymorphic code and infers/fills the generic type arguments, so the
/// monomorphizer is a pure specializer over fully-explicit type args.
fn build_module<'ctx>(ctx: &'ctx Context, src: &str, sm: &mut SourceMap) -> Result<Module<'ctx>, Vec<Diag>> {
    build_module_for(ctx, src, codegen::target_triple(), sm)
}

/// `build_module` for an explicitly chosen target triple (cross-targeting).
fn build_module_for<'ctx>(
    ctx: &'ctx Context,
    src: &str,
    triple: inkwell::targets::TargetTriple,
    sm: &mut SourceMap,
) -> Result<Module<'ctx>, Vec<Diag>> {
    build_module_dbg(ctx, src, triple, None, sm)
}

/// `build_module_for`, optionally emitting DWARF debug info (`-g`). `dbg` carries the
/// main file's `(file_name, directory)` when `-g` is on; the `DebugInput` (which also
/// needs the fully-populated `SourceMap` for its per-source `DIFile`s) is assembled
/// *after* `elaborate` has registered every imported file.
fn build_module_dbg<'ctx>(
    ctx: &'ctx Context,
    src: &str,
    triple: inkwell::targets::TargetTriple,
    dbg: Option<(String, String)>,
    sm: &mut SourceMap,
) -> Result<Module<'ctx>, Vec<Diag>> {
    let program = timed("front-end (elaborate)", || elaborate(src, sm))?;
    let program = timed("monomorphize", || mono::monomorphize(program)).map_err(|e| vec![Diag::from(e)])?;
    let dbg = dbg.map(|(main_file_name, main_directory)| codegen::DebugInput {
        sources: sm,
        main_file_name,
        main_directory,
    });
    timed("codegen (IR gen)", || codegen::compile_for_dbg(ctx, &program, triple, dbg))
        .map_err(|e| vec![Diag::from(e)])
}

/// The `(file_name, directory)` for the main file's `DIFile` (source 0), or `None`
/// when `-g` is off. `<source>` stands in when there is no real path (the library/
/// test entry points compile from a string).
fn debug_file_id(src_path: Option<&Path>) -> Option<(String, String)> {
    let p = src_path?;
    let dir = p
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());
    let name = p.file_name().map_or_else(
        || p.to_string_lossy().into_owned(),
        |n| n.to_string_lossy().into_owned(),
    );
    Some((name, dir))
}

/// Macro-expand and pretty-print the resulting forms (for `--expand`). Shows the
/// post-expansion forms (before name resolution).
pub fn expand_to_string(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    let r = (|| -> Result<Vec<macros::TaggedForm>, Diag> {
        let forms = reader::read_all(src, main)?;
        let (tagged, imports, exports) = macros::load_program(&forms, &host_target(), &mut sm)?;
        // Run Stage-3 macro expansion too, so generated definitions show up.
        expand_stage3_macros(tagged, &imports, &exports, &mut sm)
    })();
    reported(r.map_err(|d| vec![d]), &sm).map(|tagged| {
        tagged
            .iter()
            .map(|(f, _)| f.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// Canonical span-bearing dump of the reader output (`coil dump-read`) — the
/// differential-oracle target for the self-hosted reader. On a read error,
/// dumps the diagnostic in the same canonical shape (`(error@<lo>:<hi> "msg")`)
/// so error-path parity is gated too. Always `Ok`: a malformed file is a
/// well-defined dump, not a tool failure.
pub fn dump_read(src: &str) -> Result<String, String> {
    match reader::read_all(src, 0) {
        Ok(forms) => Ok(reader::dump_canonical(&forms)),
        Err(d) => {
            Ok(span::dump_diag_canonical(&d))
        }
    }
}

/// Canonical dump of the parsed program (`coil dump-ast`) — the differential
/// oracle for the self-hosted parser. Reads the (already-expanded, post-macro)
/// core forms, runs `parse::parse_program`, and dumps the whole `Program`
/// losslessly. A read OR parse error is dumped in the same canonical shape
/// (`(error@<lo>:<hi> "msg")`), so error-path parity is gated too. Always `Ok`:
/// a malformed file is a well-defined dump, not a tool failure.
pub fn dump_ast(src: &str) -> Result<String, String> {
    let parsed = reader::read_all(src, 0).and_then(|forms| parse::parse_program(&forms));
    match parsed {
        Ok(prog) => Ok(dump_ast::dump_program(&prog)),
        Err(d) => {
            Ok(span::dump_diag_canonical(&d))
        }
    }
}

/// Canonical dump of the module loader's output (`coil dump-load`) — the
/// differential oracle for the self-hosted loader. Reads the RAW (pre-macro)
/// forms, runs `macros::load_program` (which auto-loads the prelude and reads
/// every imported file from disk relative to CWD), and dumps the resulting
/// module-tagged form list + import/export tables losslessly.
///
/// Unlike `dump-read`/`dump-ast`, a load FAILURE is propagated as an error
/// (non-zero exit), not dumped canonically: the loader's failures depend on the
/// filesystem (missing imports, non-module files), so the corpus snapshot
/// excludes them by exit code rather than gating their wording.
pub fn dump_load(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    let loaded = (|| -> Result<_, Diag> {
        let forms = reader::read_all(src, main)?;
        macros::load_program(&forms, &host_target(), &mut sm)
    })();
    match loaded {
        Ok((tagged, imports, exports)) => {
            Ok(dump_load::dump_loaded(&tagged, &imports, &exports))
        }
        Err(d) => Err(span::render_all(std::slice::from_ref(&d), &sm)),
    }
}

/// Canonical dump of the Stage-3 macro EXPANDER's output (`coil dump-expand`) —
/// the differential oracle for the self-hosted macro expander. Reads the RAW
/// (pre-macro) forms, runs `macros::load_program` (auto-loading the prelude + every
/// imported file), then `expand_stage3_macros` (running the comptime macro engine to
/// a fixpoint), and dumps the resulting module-tagged form list losslessly.
///
/// A LOAD failure is propagated as an error (non-zero exit), exactly like
/// `dump-load`: the loader's failures depend on the filesystem (missing imports,
/// non-module files), so the corpus snapshot excludes them by exit code rather than
/// gating their wording. An EXPANSION failure is dumped in the canonical
/// `(error@<lo>:<hi> "msg")` shape, so error-path parity is gated too.
pub fn dump_expand(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    // Load step: a failure here is filesystem-dependent — propagate as an error so
    // the snapshot excludes it (mirrors `dump_load`).
    let loaded = (|| -> Result<_, Diag> {
        let forms = reader::read_all(src, main)?;
        macros::load_program(&forms, &host_target(), &mut sm)
    })();
    let (tagged, imports, exports) = match loaded {
        Ok(t) => t,
        Err(d) => return Err(span::render_all(std::slice::from_ref(&d), &sm)),
    };
    // Expand step: an error here is a property of the program — dump it canonically.
    match expand_stage3_macros(tagged, &imports, &exports, &mut sm) {
        Ok(expanded) => Ok(dump_expand::dump_expanded(&expanded)),
        Err(d) => {
            Ok(span::dump_diag_canonical(&d))
        }
    }
}

/// Canonical dump of the name-resolution pass's output (`coil dump-resolved`) —
/// the differential oracle for the self-hosted resolver. Reads the (already
/// post-macro / RAW core) forms, runs `macros::load_program` (auto-loading the
/// prelude + every imported file), then `resolve::resolve_program(.., strict=true)`,
/// and dumps the resulting merged, name-resolved `Program` losslessly. A read,
/// load, OR resolve error is dumped in the same canonical shape
/// (`(error@<lo>:<hi> "msg")`), so error-path parity is gated too. Always `Ok`:
/// a malformed/unresolvable file is a well-defined dump, not a tool failure.
pub fn dump_resolved(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    let r = (|| -> Result<crate::ast::Program, Diag> {
        let forms = reader::read_all(src, main)?;
        let (tagged, imports, exports) = macros::load_program(&forms, &host_target(), &mut sm)?;
        resolve::resolve_program(tagged, &imports, &exports, true)
    })();
    match r {
        Ok(p) => Ok(dump_resolved::dump_resolved(&p)),
        Err(d) => Ok(dump_resolved::dump_error(&d)),
    }
}

/// Canonical dump of the type-check pass's output (`coil dump-checked`) — the
/// differential oracle for the self-hosted checker. Reads the (post-macro / RAW
/// core) forms, runs `macros::load_program`, then `resolve::resolve_program(strict)`,
/// then `check::check_with` (with the same import/export tables), and dumps the
/// resulting typed, elaborated, lowered `Program` losslessly. A read, load,
/// resolve, OR check error is dumped in the same canonical shape
/// (`(error@<lo>:<hi> "msg")`) using the FIRST diagnostic, so error-path parity
/// is gated too. Always `Ok`: a malformed/ill-typed file is a well-defined dump.
pub fn dump_checked(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    enum E {
        One(Diag),
        Many(Vec<Diag>),
    }
    let r = (|| -> Result<crate::ast::Program, E> {
        let forms = reader::read_all(src, main).map_err(E::One)?;
        let (tagged, imports, exports) =
            macros::load_program(&forms, &host_target(), &mut sm).map_err(E::One)?;
        let resolved =
            resolve::resolve_program(tagged, &imports, &exports, true).map_err(E::One)?;
        check::check_with(&resolved, &imports, &exports).map_err(E::Many)
    })();
    match r {
        Ok(p) => Ok(dump_checked::dump_checked(&p)),
        Err(E::One(d)) => Ok(dump_checked::dump_error(&d)),
        Err(E::Many(ds)) => Ok(dump_checked::dump_error(&ds[0])),
    }
}

/// Canonical dump of the monomorphization pass's output (`coil dump-mono`) — the
/// differential oracle for the self-hosted monomorphizer. Reads the (post-macro /
/// RAW core) forms, runs `macros::load_program`, `resolve::resolve_program(strict)`,
/// `check::check_with`, then `mono::monomorphize`, and dumps the resulting
/// generics-free `Program` losslessly. A read, load, resolve, check (Diag), OR
/// mono (String) error is dumped in the same canonical shape (`(error@<lo>:<hi>
/// "msg")`); the front-end Diag uses its span, a mono error is spanless
/// (`(error@D:D "msg")`). Always `Ok`: a malformed/ill-typed file is a
/// well-defined dump, not a tool failure.
pub fn dump_mono(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    enum E {
        Diag(Diag),
        Str(String),
    }
    let r = (|| -> Result<crate::ast::Program, E> {
        let forms = reader::read_all(src, main).map_err(E::Diag)?;
        let (tagged, imports, exports) =
            macros::load_program(&forms, &host_target(), &mut sm).map_err(E::Diag)?;
        let resolved =
            resolve::resolve_program(tagged, &imports, &exports, true).map_err(E::Diag)?;
        let mut checked = check::check_with(&resolved, &imports, &exports)
            .map_err(|ds| E::Diag(ds.into_iter().next().unwrap()))?;
        // Mirror `elaborate_on_stack`: comptime-only functions (a `Code` parameter
        // or return — macros, generators, code helpers) have no runtime
        // representation and are dropped after check, before mono. Without this the
        // auto-loaded prelude's Code-returning macros reach mono and abort it.
        checked
            .funcs
            .retain(|f| f.ret != ast::Type::Code && f.params.iter().all(|p| p.ty != ast::Type::Code));
        mono::monomorphize(checked).map_err(E::Str)
    })();
    match r {
        Ok(p) => Ok(dump_mono::dump_mono(&p)),
        Err(E::Diag(d)) => Ok(dump_mono::dump_error(&d)),
        Err(E::Str(msg)) => Ok(dump_mono::dump_str_error(&msg)),
    }
}

/// Parse + check + emit textual LLVM IR (no JIT). Useful in tests and for
/// inspecting how conventions lower.
pub fn emit_ir(src: &str) -> Result<String, String> {
    let ctx = Context::create();
    let mut sm = SourceMap::new();
    let module = reported(build_module(&ctx, src, &mut sm), &sm)?;
    Ok(module.print_to_string().to_string())
}

/// `coil dump-ir` — the self-host codegen ORACLE. Emits the textual LLVM IR
/// (`emit_ir`) then runs it through `normalize_ir::normalize`, which cancels the
/// run-to-run nondeterminism in codegen's output (positional global numbering,
/// attribute-group numbering, top-level emission order) so a byte-diff gate is
/// meaningful. The self-host codegen prints its module with the same
/// `LLVMPrintModuleToString` and passes it through the same normalization, so the
/// two are compared on equal footing.
pub fn dump_ir(src: &str) -> Result<String, String> {
    Ok(normalize_ir::normalize(&emit_ir(src)?))
}

/// `coil dump-ir --target <triple>` — the normalized IR oracle for a *non-host*
/// target (e.g. the x86-64 SysV ABI from an arm64 host). Same normalization as
/// `dump_ir`, so the self-host's `emit-ir --target …` is gated on equal footing.
pub fn dump_ir_for(src: &str, triple: &str) -> Result<String, String> {
    Ok(normalize_ir::normalize(&emit_ir_for(src, triple)?))
}

/// `emit_ir` for an explicitly chosen target triple — used to inspect a
/// program's ABI lowering for a non-host target (e.g. verifying the x86-64 SysV
/// struct coercion from an arm64 host).
pub fn emit_ir_for(src: &str, triple: &str) -> Result<String, String> {
    let ctx = Context::create();
    let mut sm = SourceMap::new();
    let module = reported(
        build_module_for(&ctx, src, inkwell::targets::TargetTriple::create(triple), &mut sm),
        &sm,
    )?;
    Ok(module.print_to_string().to_string())
}

/// AOT: compile to a native object file. This is the language's primary output
/// — no runtime dependency on LLVM, links with a real linker, and the `:shim`
/// trampolines become ordinary relocations the system toolchain resolves.
pub fn compile_to_object(src: &str, obj_path: &Path) -> Result<(), String> {
    compile_to_object_for(src, obj_path, codegen::target_triple())
}

/// Place each DEFINED function in its own `.text.<name>` section
/// (`-ffunction-sections`), so a linker invoked with `--gc-sections` (the
/// freestanding recipe) can garbage-collect UNREFERENCED functions. Without this,
/// importing the stdlib drags every defn's libc calls (`malloc`/`abort`/…) into the
/// link even when unused — a freestanding program would link libc just for an
/// `import`. Done in the OBJECT path only (not `emit_ir`), so the IR text the tests
/// diff is unchanged; harmless for normal `cc` links (no `--gc-sections` → every
/// section kept), so it needs no flag — a generic mechanism, not a "freestanding mode".
/// Whether `triple` uses the ELF object format (so `.text.<name>` per-function
/// sections are valid). Mach-O (apple/darwin) and COFF (windows/msvc) have a
/// different section grammar and dead-strip differently, so they're excluded — and
/// emitting an ELF section name on Mach-O is a hard LLVM error.
fn target_uses_elf(triple: &str) -> bool {
    !(triple.contains("apple")
        || triple.contains("darwin")
        || triple.contains("macho")
        || triple.contains("windows")
        || triple.contains("msvc")
        || triple.contains("wasm"))
}

fn set_function_sections(module: &inkwell::module::Module) {
    use inkwell::values::AsValueRef;
    for f in module.get_functions() {
        if f.count_basic_blocks() == 0 {
            continue; // a declaration (extern) — no body to place in a section
        }
        if let Ok(name) = f.get_name().to_str() {
            if let Ok(sec) = std::ffi::CString::new(format!(".text.{name}")) {
                unsafe { inkwell::llvm_sys::core::LLVMSetSection(f.as_value_ref(), sec.as_ptr()) };
            }
        }
    }
}

/// `compile_to_object` for an explicitly chosen target triple. Used to
/// cross-compile (e.g. an x86-64 object on an arm64 host to exercise the SysV
/// struct ABI under Rosetta); the IR and the emitted machine code share `triple`.
pub fn compile_to_object_for(
    src: &str,
    obj_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    compile_to_object_dbg(src, obj_path, triple, None)
}

/// `compile_to_object_for`, optionally emitting DWARF (`src_path` = `Some` ⇒ `-g`,
/// using that path for the `DIFile`).
pub fn compile_to_object_dbg(
    src: &str,
    obj_path: &Path,
    triple: inkwell::targets::TargetTriple,
    src_path: Option<&Path>,
) -> Result<(), String> {
    Target::initialize_all(&InitializationConfig::default());
    let ctx = Context::create();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let dbg = debug_file_id(src_path);
    let mut sm = SourceMap::new();
    let module = reported(build_module_dbg(&ctx, src, triple, dbg, &mut sm), &sm)?;
    let triple = module.get_triple();
    // Bare-metal aarch64 (the freestanding `-none` target) runs with the MMU off, so
    // RAM is Device memory — UNALIGNED accesses fault. `+strict-align` stops the backend
    // emitting unaligned wide accesses (the optimizer otherwise uses an unaligned
    // 16-byte SIMD store for struct/array init, e.g. `stur q0,[x8,#8]`, which faults on
    // Device memory); aligned accesses (incl. an aligned vector copy) are fine. The
    // standard bare-metal approach (matches the Linux arm64 kernel). Hosted targets
    // (MMU on → Normal memory) don't need it and aren't changed.
    let triple_s = triple.as_str().to_string_lossy();
    let features = if triple_s.contains("aarch64") && triple_s.contains("none") {
        "+strict-align"
    } else {
        ""
    };
    // A `-g` build also needs the *backend* (instruction selection, scheduling,
    // store-merging) at -O0: otherwise it reorders/merges the debug-spill stores
    // away from the statement they belong to, so a local reads as stale at a line
    // breakpoint. A release build keeps Aggressive for `cc -O3` parity.
    let backend_opt = if src_path.is_some() {
        OptimizationLevel::None
    } else {
        OptimizationLevel::Aggressive
    };
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            features,
            backend_opt,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("could not create target machine")?;

    module.set_triple(&triple);
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    // Per-function sections are an ELF concept (`.text.<name>`); Mach-O and COFF use a
    // different section grammar (and dead-strip by symbol atom, not by section), so
    // only emit them for ELF targets — the freestanding/bare-metal + Linux case where
    // `--gc-sections` is used. Emitting `.text.foo` on Mach-O is a hard LLVM error.
    if target_uses_elf(&triple.as_str().to_string_lossy()) {
        set_function_sections(&module);
    }
    // Run the full LLVM optimization pipeline (mem2reg, inlining, GVN, loop
    // opts, tail-call elimination, …) before lowering to machine code. Without
    // this the emitted object would be ~`-O0`: every `let`/field stays an
    // `alloca`, nothing inlines, and self-tail-recursion (Coil's only loop) is
    // never turned into a loop (it would overflow the stack).
    //
    // A `-g` (debug) build runs an almost-empty pipeline so the code stays
    // faithful to the source for line-by-line stepping and variable inspection:
    // `alwaysinline` (the `(llvm-ir …)` zero-overhead helpers MUST still inline)
    // and `tailcallelim` (Coil's only loop is self-tail-recursion — without TRE a
    // recursive program overflows the stack). Everything else is left OFF —
    // notably mem2reg/instcombine/GVN, which would fold statements away (e.g.
    // `(iadd 1 2)` → a constant) and leave nothing to step through; keeping the
    // `alloca`s also means locals live in memory where the debugger can read them.
    let pipeline = if src_path.is_some() {
        "function(tailcallelim),always-inline"
    } else {
        OPT_PIPELINE
    };
    timed("LLVM opt pipeline", || {
        module.run_passes(pipeline, &tm, PassBuilderOptions::create())
    })
    .map_err(|e| format!("optimization passes failed: {e}"))?;
    timed("object emit", || tm.write_to_file(&module, FileType::Object, obj_path))
        .map_err(|e| e.to_string())
}

/// AOT: compile to an object and link a native executable with `cc`. The Coil
/// `main` (i64, no args) becomes the process entry; its return value is the
/// exit code (low 8 bits).
pub fn build_executable(src: &str, out_path: &Path) -> Result<(), String> {
    build_executable_for(src, out_path, codegen::target_triple())
}

/// `build_executable` for an explicitly chosen target triple (cross-compile).
/// The object is emitted for `triple` *and* the linker is told to produce an
/// executable for that triple's architecture (`cc -arch <arch>`), so the result
/// is a real binary for the chosen target (runnable under Rosetta on macOS for a
/// cross arch). The triple's arch must be one codegen supports.
pub fn build_executable_for(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    build_executable_linked(src, out_path, triple, &[])
}

/// `build_executable_for` plus extra arguments passed through to the `cc` link line —
/// e.g. `-lm`, `-lfoo`, or a C object file path — so a Coil program can link against C
/// libraries / objects (the C-interop §6 linking half). A generic passthrough, NOT a
/// baked-in "C-lib mode": the compiler emits the object; how it's linked is the caller's.
pub fn build_executable_linked(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    link_flags: &[String],
) -> Result<(), String> {
    build_executable_linked_dbg(src, out_path, triple, link_flags, None)
}

/// `build_executable_linked`, optionally emitting DWARF. `src_path = Some` turns
/// on debug info (`-g`) and names the `DIFile`. On macOS the DWARF stays in the
/// `.o` (the linker records a debug map that points at it), so a debug build
/// keeps the object file and runs `dsymutil` to gather a `.dSYM` next to the
/// executable; a release build deletes the `.o` as before.
pub fn build_executable_linked_dbg(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    link_flags: &[String],
    src_path: Option<&Path>,
) -> Result<(), String> {
    let triple_str = triple.as_str().to_string_lossy().into_owned();
    let obj_path = out_path.with_extension("o");
    compile_to_object_dbg(src, &obj_path, triple, src_path)?;
    let mut cc = Command::new("cc");
    if let Some(arch) = link_arch_flag(&triple_str) {
        cc.arg("-arch").arg(arch);
    }
    cc.arg(&obj_path).arg("-o").arg(out_path);
    for f in link_flags {
        cc.arg(f);
    }
    let status = timed("link (cc)", || cc.status())
        .map_err(|e| format!("failed to invoke linker (cc): {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&obj_path);
        return Err(format!("linker (cc) failed with {status}"));
    }
    if src_path.is_some() {
        // macOS keeps DWARF in the `.o`; collect it into a `.dSYM` so the debugger
        // finds it even after the `.o` is gone. Only remove the `.o` if dsymutil
        // actually succeeded — otherwise keep it as the debug-map fallback (losing
        // both would silently strip all debug info).
        let dsym_ok = Command::new("dsymutil")
            .arg(out_path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if dsym_ok {
            let _ = std::fs::remove_file(&obj_path);
        }
    } else {
        let _ = std::fs::remove_file(&obj_path);
    }
    Ok(())
}

/// The `cc -arch` value for a target triple's architecture, or `None` if it is
/// the host arch (no `-arch` needed). An unsupported arch is a hard error so a
/// `--target` typo doesn't silently link for the host.
fn link_arch_flag(triple: &str) -> Option<&'static str> {
    let arch = triple.split('-').next().unwrap_or("");
    let host = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .split('-')
        .next()
        .unwrap_or("")
        .to_string();
    match arch {
        "x86_64" | "amd64" => (host != "x86_64").then_some("x86_64"),
        "aarch64" | "arm64" | "arm64e" => (host != "aarch64" && host != "arm64").then_some("arm64"),
        _ => None,
    }
}

/// Generate a C header for a program's `(export-c …)` set — function prototypes
/// under their C symbols plus the structs reachable from those signatures. The
/// inverse of `cimport`. See `docs/SYMBOL_EXPORT.md`.
pub fn emit_header(src: &str) -> Result<String, String> {
    let mut sm = SourceMap::new();
    let program = reported(elaborate(src, &mut sm), &sm)?;
    cheader::render(&program)
}

/// AOT to a STATIC LIBRARY (`.a`): compile to an object and archive it with `ar`.
/// A library has no `main`; its `export-c` functions are the public symbols (the
/// rest internalize). The object is consumed into the archive and removed.
pub fn build_static_lib(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    src_path: Option<&Path>,
) -> Result<(), String> {
    let obj_path = out_path.with_extension("o");
    compile_to_object_dbg(src, &obj_path, triple, src_path)?;
    let status = Command::new("ar")
        .arg("rcs")
        .arg(out_path)
        .arg(&obj_path)
        .status()
        .map_err(|e| format!("failed to invoke archiver (ar): {e}"))?;
    let _ = std::fs::remove_file(&obj_path);
    if !status.success() {
        return Err(format!("archiver (ar) failed with {status}"));
    }
    Ok(())
}

/// AOT to a SHARED LIBRARY (`.dylib`/`.so`): compile to an object and link it as a
/// dynamic library with `cc` (`-dynamiclib` on Apple, `-shared` elsewhere). Extra
/// `link_flags` (e.g. `-lm`) are passed through.
pub fn build_shared_lib(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    link_flags: &[String],
    src_path: Option<&Path>,
) -> Result<(), String> {
    let triple_str = triple.as_str().to_string_lossy().into_owned();
    let obj_path = out_path.with_extension("o");
    compile_to_object_dbg(src, &obj_path, triple, src_path)?;
    let mut cc = Command::new("cc");
    if let Some(arch) = link_arch_flag(&triple_str) {
        cc.arg("-arch").arg(arch);
    }
    let apple = triple_str.contains("apple") || triple_str.contains("darwin") || triple_str.contains("macos");
    cc.arg(if apple { "-dynamiclib" } else { "-shared" });
    cc.arg(&obj_path).arg("-o").arg(out_path);
    for f in link_flags {
        cc.arg(f);
    }
    let status = cc.status().map_err(|e| format!("failed to invoke linker (cc): {e}"))?;
    let _ = std::fs::remove_file(&obj_path);
    if !status.success() {
        return Err(format!("linker (cc) failed with {status}"));
    }
    Ok(())
}

/// Front end only: read → expand → parse → check. No codegen, no LLVM
/// execution — just diagnostics. (Coil has no `eval`/JIT: the only way to run a
/// program is to AOT-compile it.)
/// REPL support (`coil repl`): run the front end on `src` and infer the type of
/// the tail expression of the function named `probe`, with no expected type. No
/// whole-program type-check runs — the REPL fully checks every definition as it
/// enters the session, so only the probe body needs synthesis here (its errors
/// carry the normal caret diagnostics). Returns the inferred type together with
/// the resolved program, whose struct/sum tables drive the REPL's value display.
pub fn repl_infer_tail(src: &str, probe: &str) -> Result<(ast::Type, ast::Program), String> {
    let mut sm = SourceMap::new();
    let main = sm.add("<source>", src);
    // Same big-stack discipline as `elaborate`: the front end + `synth` recurse
    // over expression nesting (and always pull in the prelude's macro engine).
    let r = std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn_scoped(s, || -> Result<(ast::Type, ast::Program), Vec<Diag>> {
                let (program, imports, exports) = front_end_on_stack(src, main, &mut sm)?;
                let ty = one(check::infer_tail_type(&program, &imports, &exports, probe))?;
                Ok((ty, program))
            })
            .expect("spawn front-end thread")
            .join()
            .expect("front-end thread panicked")
    });
    reported(r, &sm)
}

pub fn check_source(src: &str) -> Result<(), String> {
    let mut sm = SourceMap::new();
    let program = reported(elaborate(src, &mut sm), &sm)?;
    // Run monomorphization too, so specialization-time errors (if any) surface
    // in a check-only pass as well.
    reported(mono::monomorphize(program).map_err(|e| vec![Diag::from(e)]), &sm)?;
    Ok(())
}
