//! Source-span diagnostics: reader, parser, AND type-checker errors render
//! `file:line:col`, the offending source line, and a caret. Checker errors carry
//! spans now too (every `Expr` carries its source span; the `synth` recursion
//! attaches the innermost offending expression's span to the error).

/// A parse-level arity error points at the offending form with a caret.
#[test]
fn parse_error_has_location_and_caret() {
    let src = "(defn main [] (-> :i64)\n  (iadd 1))\n";
    let err = coil::emit_ir(src).unwrap_err();
    assert!(err.contains("expects exactly 2 arguments"), "got:\n{err}");
    // Rendered against the library's `<source>` placeholder; the bad form is on
    // line 2, column 3.
    assert!(err.contains("<source>:2:3"), "missing file:line:col:\n{err}");
    assert!(err.contains('^'), "missing caret:\n{err}");
    assert!(err.contains("(iadd 1))"), "missing source line:\n{err}");
}

/// A reader-level unclosed delimiter is located, not a bare message.
#[test]
fn reader_error_has_location() {
    let src = "(defn main [] (-> :i64)\n  (foo 1 2)\n";
    let err = coil::emit_ir(src).unwrap_err();
    assert!(err.contains("unclosed"), "got:\n{err}");
    assert!(err.contains("-->"), "reader error should be located:\n{err}");
}

/// A bad type name in a value position is located at that form.
#[test]
fn bad_type_is_located() {
    let src = "(defn main [] (-> :i64)\n  (sizeof 7))\n";
    let err = coil::emit_ir(src).unwrap_err();
    assert!(err.contains("-->"), "should be located:\n{err}");
    assert!(err.contains(":2:"), "should point at line 2:\n{err}");
}

/// An unbound-variable checker error now points at the offending sub-expression
/// with a caret (not a bare message).
#[test]
fn checker_unbound_var_is_located() {
    let src = "(defn main [] (-> :i64)\n  (iadd x 1))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("unbound variable 'x'"), "got:\n{err}");
    assert!(err.contains("-->"), "checker error should be located:\n{err}");
    assert!(err.contains(":2:"), "should point at line 2:\n{err}");
    assert!(err.contains('^'), "missing caret:\n{err}");
    assert!(err.contains("(iadd x 1))"), "missing source line:\n{err}");
}

/// A type-mismatch checker error is located at the offending expression — the
/// innermost span wins (the `(iadd …)` form, on line 2), not the whole function.
#[test]
fn checker_type_mismatch_is_located() {
    let src = "(defn main [] (-> i64)\n  (iadd 1 2.0))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("different types"), "got:\n{err}");
    assert!(err.contains("-->") && err.contains(":2:"), "should be located at line 2:\n{err}");
    assert!(err.contains('^'), "missing caret:\n{err}");
}

/// Multi-error: independent type errors in different functions are ALL reported in
/// one pass (the checker collects per-function errors instead of stopping at the
/// first), each with its own location, plus an `N errors` summary.
#[test]
fn checker_reports_multiple_errors_at_once() {
    let src = "(module app)\n\
               (defn f [(x i64)] (-> i64) (iadd x 1.5))\n\
               (defn g [(y i64)] (-> i64) (iadd y 2.5))\n\
               (defn main [] (-> i64) (iadd (f 1) (g 2)))\n";
    let err = coil::check_source(src).unwrap_err();
    // Both functions' errors must appear, each located on its own line.
    assert!(err.contains("app.f"), "missing f's error:\n{err}");
    assert!(err.contains("app.g"), "missing g's error:\n{err}");
    assert!(err.contains(":2:") && err.contains(":3:"), "both lines located:\n{err}");
    assert!(err.contains("2 errors"), "missing error count summary:\n{err}");
}

/// A single error does NOT get an `N errors` summary (it reads as a plain one).
#[test]
fn single_error_has_no_count_summary() {
    let src = "(defn main [] (-> i64)\n  (iadd 1 2.0))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(!err.contains("errors"), "single error should have no count:\n{err}");
}

/// Macro provenance: a type error introduced by a macro's *template* renders with a
/// caret at the call site PLUS an `in expansion of macro …` note pointing at the
/// macro's definition (the gold-standard expansion trace).
#[test]
fn macro_template_error_shows_expansion_trace() {
    // line 2: the macro; line 3: the call. The template injects an f64 (`1.5`) next
    // to the user's i64 arg, so `(iadd <i64> 1.5)` is a type error in generated code.
    let src = "(module app)\n\
               (defn add-half [(x Code)] (-> Code) `(iadd ~x 1.5))\n\
               (defn main [] (-> i64) (add-half 3))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("different types"), "got:\n{err}");
    assert!(err.contains("in expansion of macro `app.add-half`"), "missing trace:\n{err}");
    // Primary caret at the call site (line 3); note at the macro definition (line 2).
    assert!(err.contains(":3:") && err.contains(":2:"), "call site + def site:\n{err}");
}

/// No mis-attribution: a type error in a user expression *passed through* a macro
/// (spliced via `~unquote`) is located at the USER's code and carries NO macro
/// expansion trace — the spliced node keeps the caller's span, not the macro's.
#[test]
fn user_spliced_error_is_not_attributed_to_the_macro() {
    let src = "(module app)\n\
               (defn wrap [(e Code)] (-> Code) `(do ~e))\n\
               (defn main [] (-> i64) (wrap (iadd 1 2.0)))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("different types"), "got:\n{err}");
    assert!(!err.contains("in expansion of"), "user code must NOT show a macro trace:\n{err}");
    assert!(err.contains(":3:"), "located at the user's call on line 3:\n{err}");
}
