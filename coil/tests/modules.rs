//! Module system: `(module …)` + `(import … :as …)` namespaces function
//! definitions so different modules can define the same name without colliding,
//! and imports resolve relative to the importing file (not the cwd).

mod common;
use common::build_and_run;

/// Write `files` (name -> contents) into a fresh temp dir; return the dir.
fn write_modules(tag: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("coil_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    dir
}

/// Robust hygiene: a macro in module A referencing a *macro-generated* definition
/// of A still resolves in A when used from another module (incremental namespace
/// building — A is fully expanded before B's macros run).
#[test]
fn second_order_macro_hygiene_across_modules() {
    let dir = write_modules(
        "mod_so",
        &[(
            "gen.coil",
            "(module gen)\n\
             (defn def-secret [] (-> Code) `(defn secret [(x :i64)] (-> :i64) (iadd x 100)))\n\
             (def-secret)\n\
             (defn bump [(v Code)] (-> Code) `(secret ~v))\n",
        )],
    );
    let src = format!(
        "(module app)\n(import \"{}\" :use *)\n(defn secret [(x :i64)] (-> :i64) x)\n\
         (defn main [] (-> :i64) (bump 5))\n", // gen's macro-generated secret -> 105, not app's 5
        dir.join("gen.coil").display()
    );
    assert_eq!(build_and_run(&src), 105);
}

/// `(export …)` makes only the listed names visible to other modules; an
/// exported name works, own-module access to a private name still works.
#[test]
fn export_allows_listed_names() {
    let dir = write_modules(
        "mod_exp",
        &[(
            "lib.coil",
            "(module lib)\n(export pub-fn)\n\
             (defn pub-fn [(x :i64)] (-> :i64) (helper x))\n\
             (defn helper [(x :i64)] (-> :i64) (iadd x 1))\n", // private, used internally
        )],
    );
    let src = format!(
        "(module app)\n(import \"{}\" :as l)\n(defn main [] (-> :i64) (l/pub-fn 41))\n", // 42
        dir.join("lib.coil").display()
    );
    assert_eq!(build_and_run(&src), 42);
}

/// Accessing a non-exported name across modules is a hard error.
#[test]
fn export_hides_private_names() {
    let dir = write_modules(
        "mod_priv",
        &[(
            "lib.coil",
            "(module lib)\n(export pub-fn)\n\
             (defn pub-fn [(x :i64)] (-> :i64) x)\n\
             (defn helper [(x :i64)] (-> :i64) (iadd x 1))\n",
        )],
    );
    let src = format!(
        "(module app)\n(import \"{}\" :as l)\n(defn main [] (-> :i64) (l/helper 41))\n",
        dir.join("lib.coil").display()
    );
    let err = coil::check_source(&src).unwrap_err();
    assert!(err.contains("private"), "expected a privacy error, got: {err}");
}

/// Two modules each define `sq`; the importer also defines its own `sq`. All
/// three coexist (namespaced to `mathx/sq`, `app/sq`), and a cross-module call
/// goes through the `:as` alias. `main` stays the bare entry point.
#[test]
fn two_modules_same_function_name() {
    let dir = std::env::temp_dir().join(format!("coil_mod_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mathx = dir.join("mathx.coil");
    std::fs::write(&mathx, "(module mathx)\n(defn sq [(x :i64)] (-> :i64) (imul x x))\n").unwrap();

    // absolute import path so the test is independent of the working directory
    let src = format!(
        "(module app)\n\
         (import \"{}\" :as m)\n\
         (defn sq [(x :i64)] (-> :i64) (iadd x 1))\n\
         (defn main [] (-> :i64) (iadd (m/sq 6) (sq 5)))\n", // 36 + 6 = 42
        mathx.display()
    );
    assert_eq!(build_and_run(&src), 42);
}

/// A module calling its *own* functions unqualified still resolves (recursion
/// and intra-module calls get namespaced consistently).
#[test]
fn intra_module_calls_resolve() {
    let dir = std::env::temp_dir().join(format!("coil_mod2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lib = dir.join("lib.coil");
    std::fs::write(
        &lib,
        "(module lib)\n\
         (defn dbl [(x :i64)] (-> :i64) (imul x 2))\n\
         (defn quad [(x :i64)] (-> :i64) (dbl (dbl x)))\n", // own-module call
    )
    .unwrap();
    let src = format!(
        "(module app)\n(import \"{}\" :as l)\n(defn main [] (-> :i64) (l/quad 10))\n", // 40
        lib.display()
    );
    assert_eq!(build_and_run(&src), 40);
}

/// Macro hygiene across modules: a macro defined in module A that references A's
/// own function resolves to A — even when the use site defines a *different*
/// function of the same name (Clojure's syntax-quote rule).
#[test]
fn macro_references_resolve_in_defining_module() {
    let dir = std::env::temp_dir().join(format!("coil_mod4_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let m = dir.join("mathmod.coil");
    std::fs::write(
        &m,
        "(module mathmod)\n\
         (defn secret [(x :i64)] (-> :i64) (iadd x 100))\n\
         (defn bump [(v Code)] (-> Code) `(secret ~v))\n", // references mathmod's own `secret`
    )
    .unwrap();
    let src = format!(
        "(module app)\n\
         (import \"{}\" :use *)\n\
         (defn secret [(x :i64)] (-> :i64) x)\n\
         (defn main [] (-> :i64) (bump 5))\n", // 105 (mathmod) not 5 (app)
        m.display()
    );
    assert_eq!(build_and_run(&src), 105);
}

/// Cross-module *types* and *sums*: a struct `Point` defined in two modules
/// coexists; a `:as` alias reaches a cross-module struct; and `:use *` brings a
/// generic sum's constructors/functions in unqualified.
#[test]
fn cross_module_types_sums_and_use() {
    let dir = std::env::temp_dir().join(format!("coil_mod3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let opt = dir.join("opt.coil");
    std::fs::write(
        &opt,
        "(module opt)\n\
         (defsum Maybe [T] (Nothing) (Just [(val T)]))\n\
         (defn unwrap [T] [(o (Maybe T)) (d T)] (-> T) (match o (Nothing [] d) (Just [v] v)))\n",
    )
    .unwrap();
    let geom = dir.join("geom.coil");
    std::fs::write(
        &geom,
        "(module geom)\n\
         (defstruct Point [(x :i64) (y :i64)])\n\
         (defn sumxy [(p (ptr Point))] (-> :i64) (iadd (load (field p x)) (load (field p y))))\n",
    )
    .unwrap();
    let src = format!(
        "(module app)\n\
         (import \"{}\" :use *)\n\
         (import \"{}\" :as g)\n\
         (defstruct Point [(a :i64)])\n\
         (defn main [] (-> :i64)\n\
           (let [p (alloc-stack g/Point)]\n\
             (store! (field p x) 30) (store! (field p y) 12)\n\
             (iadd (g/sumxy p) (unwrap [i64] (Just [i64] 0) 0))))\n", // 42 + 0
        opt.display(),
        geom.display()
    );
    assert_eq!(build_and_run(&src), 42);
}

#[test]
fn importing_the_same_file_twice_is_idempotent() {
    // The loader's `visited` dedup: importing the identical path twice in one
    // program must not double-define its functions (no "duplicate definition").
    let dir = write_modules(
        "mod_dup",
        &[("util.coil", "(module util)\n(defn u [(x :i64)] (-> :i64) (iadd x 7))\n")],
    );
    let p = dir.join("util.coil");
    let src = format!(
        "(module app)\n(import \"{0}\" :use *)\n(import \"{0}\" :use *)\n\
         (defn main [] (-> :i64) (u 35))\n",
        p.display()
    );
    assert_eq!(build_and_run(&src), 42);
}

#[test]
fn reflection_on_an_ambiguous_cross_module_type_errors() {
    // A bare type name that two imported modules both define must be a HARD ERROR
    // in field reflection, never a silent pick of the wrong struct (regression:
    // a derived eq once read 5 fields off a 1-field struct).
    let dir = write_modules(
        "mod_amb",
        &[("other.coil", "(module other)\n(export P)\n(defstruct P [(a i64) (b i64) (c i64)])\n")],
    );
    let src = format!(
        "(module app)\n(import \"lib/derive.coil\" :use *)\n(import \"{}\" :use *)\n\
         (defstruct P [(v i64)])\n(derive Eq P)\n(defn main [] (-> :i64) 0)\n",
        dir.join("other.coil").display()
    );
    let err = coil::check_source(&src).unwrap_err();
    assert!(err.contains("ambiguous"), "expected an ambiguity error, got:\n{err}");
}

#[test]
fn macros_are_namespaced_not_global() {
    // Clojure-like: `import` alone does NOT refer a module's macros — a bare call is
    // undefined until you `:use` (refer) it. Uses a custom (non-core) macro so it
    // tests scoping, not coil.core's re-exports.
    let dir = write_modules(
        "mod_macns",
        &[("m.coil", "(module m)\n(defn twice [(x Code)] (-> Code) `(iadd ~x ~x))\n")],
    );
    let p = dir.join("m.coil");
    let bare = format!(
        "(module app)\n(import \"{}\")\n(defn main [] (-> :i64) (twice 21))\n",
        p.display()
    );
    let err = coil::check_source(&bare).unwrap_err();
    assert!(err.contains("undefined") && err.contains("twice"), "got: {err}");
    // …but `:use *` (refer all) brings it in, and `:as` gives qualified access.
    let used = format!(
        "(module app)\n(import \"{}\" :use *)\n(defn main [] (-> :i64) (twice 21))\n",
        p.display()
    );
    assert_eq!(build_and_run(&used), 42);
    let aliased = format!(
        "(module app)\n(import \"{}\" :as m)\n(defn main [] (-> :i64) (m/twice 21))\n",
        p.display()
    );
    assert_eq!(build_and_run(&aliased), 42);
}


#[test]
fn trait_methods_are_namespaced_per_module() {
    // Two modules each define a `Show` trait with a `show` method — no longer a
    // global-uniqueness error. A call resolves to whichever is `:use`d; `:use`ing
    // both makes a bare call ambiguous (a hard error).
    let dir = write_modules(
        "mod_meth",
        &[
            ("a.coil", "(module a)\n(export Show show A)\n\
                        (deftrait Show [Self] (show [(x Self)] (-> :i64)))\n\
                        (defstruct A [(v :i64)])\n\
                        (impl Show A (show [(x A)] (-> :i64) (iadd (load (field x v)) 100)))\n"),
            ("b.coil", "(module b)\n(export Show show)\n\
                        (deftrait Show [Self] (show [(x Self)] (-> :i64)))\n"),
        ],
    );
    let mk = |imports: &str| {
        format!(
            "(module app)\n{imports}\n\
             (defn run [(n :i64)] (-> :i64) (let [(mut x) (zeroed A)] (store! (field (mut x) v) n) (show (load x))))\n\
             (defn main [] (-> :i64) (run 5))\n",
        )
    };
    // Only a's Show referred → resolves to a's show.
    let one = format!("(import \"{}\" :use *)", dir.join("a.coil").display());
    assert_eq!(build_and_run(&mk(&one)), 105);
    // Both referred → ambiguous.
    let both = format!(
        "(import \"{}\" :use *)\n(import \"{}\" :use *)",
        dir.join("a.coil").display(),
        dir.join("b.coil").display()
    );
    let err = coil::check_source(&mk(&both)).unwrap_err();
    assert!(err.contains("ambiguous"), "expected ambiguity, got: {err}");
}

#[test]
fn undefined_call_is_a_resolve_error() {
    // Stage 4: a bare callee that is neither defined, an extern, nor a trait method
    // is reported as undefined (no silent fallthrough).
    let err = coil::check_source("(module app)\n(defn main [] (-> :i64) (no-such-fn 1))\n").unwrap_err();
    assert!(err.contains("undefined function 'no-such-fn'"), "got: {err}");
}
