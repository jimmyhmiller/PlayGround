//! Dogfood: the tiny expression-language interpreter (examples/calc.coil) — the
//! Phase-1 capstone exercising ArrayList + a recursive AST sum + HashMap + fmt +
//! control flow + explicit allocator, all together. We compile+run the actual
//! example and check its stdout (precedence, parens, assignment, variable lookup).

mod common;
use common::build_and_capture;

#[test]
fn calc_example_produces_expected_output() {
    let src = std::fs::read_to_string("examples/calc.coil")
        .expect("read examples/calc.coil");
    let (_code, out) = build_and_capture(&src);
    // 1+2=3 ; 3+2*5=13 (precedence) ; (3+10)*2=26 (parens) ; x=27 ; x=27 (HashMap env)
    assert_eq!(out, "3\n13\n26\n27\n27\n");
}
