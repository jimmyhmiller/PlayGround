//! Source-span diagnostics: reader/parser errors render `file:line:col`, the
//! offending source line, and a caret. (Checker-level errors don't carry spans
//! yet — that's the next slice — so they stay bare messages, asserted here so the
//! staged boundary is explicit.)

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

/// Checker errors stay bare (no span) for now — they must still surface their
/// message and must NOT crash the renderer.
#[test]
fn checker_error_is_bare_message_for_now() {
    let src = "(defn main [] (-> :i64)\n  (iadd x 1))\n";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("unbound variable 'x'"), "got:\n{err}");
    // No span ⇒ no caret/location frame.
    assert!(!err.contains("-->"), "checker errors shouldn't fabricate a span yet:\n{err}");
}
