//! Round-trip tests for break/continue (real CFG edges to loop exit/continue).

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn rt(src: &str) {
    let op = jsir_swc::source_to_ir(src).unwrap();
    let direct = jsir_swc::ir_to_source(&op).unwrap();
    let (lifted, _) = jsir_jslir::roundtrip(&op);
    let ours = jsir_swc::ir_to_source(&lifted).unwrap();
    assert_eq!(norm(&direct), norm(&ours), "diverged:\n{src}");
}
fn lowered(src: &str) -> usize {
    let op = jsir_swc::source_to_ir(src).unwrap();
    jsir_jslir::build_jslir(&op).1.lowered
}

#[test]
fn break_in_while() {
    rt("function f(x) { while (x) { if (x) { break; } g(); } }");
    assert_eq!(lowered("function f(x) { while (x) { if (x) { break; } g(); } }"), 1);
}

#[test]
fn continue_in_while() {
    rt("function f(x) { while (x) { if (x) { continue; } g(); } }");
    assert_eq!(lowered("function f(x) { while (x) { if (x) { continue; } g(); } }"), 1);
}

#[test]
fn break_in_for() {
    rt("function f(n) { for (let i = 0; i < n; i = i + 1) { if (i) { break; } g(i); } }");
    assert_eq!(
        lowered("function f(n) { for (let i = 0; i < n; i = i + 1) { if (i) { break; } g(i); } }"),
        1
    );
}

#[test]
fn continue_in_for() {
    rt("function f(n) { for (let i = 0; i < n; i = i + 1) { if (i) { continue; } g(i); } }");
}

#[test]
fn unconditional_break() {
    rt("function f(x) { while (x) { g(); break; } }");
}

#[test]
fn break_and_continue_mixed() {
    rt("function f(x) { while (x) { if (x) { break; } if (x) { continue; } h(); } }");
}

#[test]
fn labeled_break_passes_through() {
    // Labeled break sits inside a labeled_statement → whole function passes through.
    let src = "function f(x) { outer: while (x) { break outer; } }";
    rt(src);
    assert_eq!(lowered(src), 0);
}
