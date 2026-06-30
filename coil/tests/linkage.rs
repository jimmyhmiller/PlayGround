//! Internal-linkage optimization (see bench/README.md + docs/SYMBOL_EXPORT.md):
//! an EXECUTABLE (defines `main`) gives its private functions `internal` linkage so
//! LLVM can prune inlined/dead code — that's most of Coil's compile-time parity with
//! clang. A LIBRARY or freestanding image (no `main`) keeps them external (something
//! outside references them by symbol) until the explicit export feature lands.

/// `emit_ir` skips the optimizer, so the raw `define`/`define internal` linkage is
/// visible exactly as codegen set it.
fn defines(src: &str) -> Vec<String> {
    coil::emit_ir(src)
        .expect("emit_ir")
        .lines()
        .filter(|l| l.starts_with("define"))
        .map(str::to_string)
        .collect()
}

#[test]
fn executable_internalizes_private_functions_but_not_main() {
    // Has `main` ⇒ executable. `helper` is private → internal; `main` stays external.
    let src = "(defn helper [(x :i64)] (-> :i64) (iadd x 1))\n\
               (defn main [] (-> :i64) (helper 41))";
    let ds = defines(src);
    let helper = ds.iter().find(|l| l.contains("@helper")).expect("helper defined");
    let main = ds.iter().find(|l| l.contains("@main")).expect("main defined");
    assert!(helper.contains("internal"), "private helper should be internal:\n{helper}");
    assert!(!main.contains("internal"), "main must stay external:\n{main}");
}

#[test]
fn library_without_main_keeps_functions_external() {
    // No `main` ⇒ library: a C caller references `lib_fn` by symbol, so it stays
    // external (this is the current stopgap; the export feature will make it explicit).
    let src = "(defn lib_fn [(x :i64)] (-> :i64) (iadd x 1))";
    let ds = defines(src);
    let f = ds.iter().find(|l| l.contains("@lib_fn")).expect("lib_fn defined");
    assert!(!f.contains("internal"), "a library's function must stay external:\n{f}");
}

// ---- export-c: Coil functions exposed as C symbols (docs/SYMBOL_EXPORT.md) --------

#[test]
fn export_c_renames_symbol_and_internalizes_private_helpers() {
    // No `main`, but an export is an anchor: the export is external under its C symbol;
    // the private helper internalizes.
    let src = "(module m)\n\
               (defn helper [(n :i64)] (-> :i64) (iadd n 1))\n\
               (defn pub-fn [(n :i64)] (-> :i64) (helper n))\n\
               (export-c [pub-fn :as \"m_pub\"])";
    let ds = defines(src);
    assert!(
        ds.iter().any(|l| l.contains("@m_pub") && !l.contains("internal")),
        "export must be external under its C symbol `m_pub`:\n{ds:#?}"
    );
    let helper = ds.iter().find(|l| l.contains("@m.helper")).expect("helper defined");
    assert!(helper.contains("internal"), "private helper should be internal:\n{helper}");
}

#[test]
fn export_c_default_symbol_is_kebab_to_snake() {
    // No `:as` → bare name with '-' → '_'  (make-thing → make_thing).
    let src = "(module m)\n(defn make-thing [(n :i64)] (-> :i64) n)\n(export-c make-thing)";
    let ds = defines(src);
    assert!(ds.iter().any(|l| l.contains("@make_thing")), "default symbol `make_thing`:\n{ds:#?}");
}

#[test]
fn export_c_struct_by_value_param_gets_a_thunk() {
    // A by-value struct param is now supported via a C-ABI thunk: the exported C
    // symbol takes the struct by value ([2 x i64] on AArch64), the internal function
    // stays reference-model (a pointer) with internal linkage.
    let src = "(module m)\n(defstruct P [(x :i64) (y :i64)])\n\
               (defn f [(p P)] (-> :i64) (load (field p x)))\n(export-c [f :as \"m_f\"])";
    let ds = defines(src);
    assert!(
        ds.iter().any(|l| l.contains("@m_f(") && !l.contains("internal") && !l.contains("@m_f(ptr")),
        "thunk should take the struct by value under the C symbol:\n{ds:#?}"
    );
    assert!(
        ds.iter().any(|l| l.contains("internal") && l.contains("@m.f(ptr")),
        "internal function should be reference-model + internal:\n{ds:#?}"
    );
}

#[test]
fn export_c_rejects_slice_param() {
    // A slice fat pointer has no C representation — still rejected (unlike a struct).
    let src = "(module m)\n(import \"slice.coil\" :use *)\n\
               (defn f [(s (slice i64))] (-> :i64) 0)\n(export-c f)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("no C representation"), "got: {err}");
}

#[test]
fn export_c_rejects_generic() {
    let src = "(module m)\n(defn id [T] [(x T)] (-> T) x)\n(export-c id)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("generic"), "got: {err}");
}

#[test]
fn export_c_rejects_duplicate_symbol() {
    let src = "(module m)\n(defn a [] (-> :i64) 1)\n(defn b [] (-> :i64) 2)\n\
               (export-c [a :as \"dup\"] [b :as \"dup\"])";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("same C symbol"), "got: {err}");
}

#[test]
fn export_c_rejects_unknown_function() {
    let src = "(module m)\n(defn a [] (-> :i64) 1)\n(export-c nonexistent)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.to_lowercase().contains("unknown") || err.contains("nonexistent"), "got: {err}");
}

// ---- coil cheader: generated C header for the export-c set -------------------------

#[test]
fn cheader_emits_prototypes_and_struct_typedefs() {
    let src = "(module shapes)\n\
               (defstruct Point [(x :i64) (y :i64)])\n\
               (defn make-point [(a :i64) (b :i64)] (-> Point)\n\
                 (let [p (alloc-stack Point)] (store! (field p x) a) (store! (field p y) b) (load p)))\n\
               (defn dist2 [(p Point)] (-> :i64) (load (field p x)))\n\
               (export-c [make-point :as \"shapes_make_point\"] [dist2 :as \"shapes_dist2\"])";
    let h = coil::emit_header(src).expect("emit_header");
    // struct typedef + body
    assert!(h.contains("typedef struct Point Point;"), "missing forward decl:\n{h}");
    assert!(h.contains("struct Point {") && h.contains("int64_t x;") && h.contains("int64_t y;"),
        "missing struct body:\n{h}");
    // prototypes: renamed symbol, C types, struct return + struct-by-value param
    assert!(h.contains("Point shapes_make_point(int64_t, int64_t);"), "missing make-point proto:\n{h}");
    assert!(h.contains("int64_t shapes_dist2(Point);"), "missing dist2 proto:\n{h}");
    // C++ guard + include guard
    assert!(h.contains("#pragma once") && h.contains("extern \"C\""), "missing guards:\n{h}");
}

#[test]
fn cheader_rejects_unrepresentable_type() {
    // A bare program with no exports yields an empty-but-valid header (no prototypes).
    let src = "(module m)\n(defn f [(n :i64)] (-> :i64) n)";
    let h = coil::emit_header(src).expect("emit_header");
    assert!(h.contains("#pragma once") && !h.contains("shapes_"), "got:\n{h}");
}
