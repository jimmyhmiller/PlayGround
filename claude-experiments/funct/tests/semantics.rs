//! Adversarial language-semantics coverage: the spots most likely to be subtly
//! wrong in a young compiler/VM — closure capture, or-patterns, range bounds,
//! guards, `?` in odd positions, empty-collection ops, and unicode slicing.

use funct::{Funct, Value};

fn eval(src: &str) -> Value {
    let mut vm = Funct::new();
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed: {}\nsource:\n{}", e, src))
}

fn s(x: &str) -> Value {
    Value::str(x)
}

#[test]
fn loop_variable_is_captured_per_iteration() {
    // Each closure must close over its OWN `i`, not a single shared slot.
    let v = eval(
        "fn make() {\n let mut fns = []\n for i in 0..3 { fns = fns.push(() => i) }\n fns\n}\nmake() |> map(f => f())",
    );
    assert_eq!(
        v,
        Value::list(vec![Value::Int(0), Value::Int(1), Value::Int(2)])
    );
}

#[test]
fn closures_share_one_captured_mutable_cell() {
    // Two closures over the same `let mut` must see each other's writes.
    let v = eval(
        "fn make() {\n let mut n = 0\n let inc = () => { n = n + 1; n }\n let get = () => n\n (inc, get)\n}\nlet (inc, get) = make()\ninc()\ninc()\nget()",
    );
    assert_eq!(v, Value::Int(2));
}

#[test]
fn or_pattern_shares_a_binding() {
    let src = "fn either(x) = match x { Ok(n) | Err(n) => n }\n(either(Ok(5)), either(Err(9)))";
    assert_eq!(eval(src), Value::tuple(vec![Value::Int(5), Value::Int(9)]));
}

#[test]
fn range_patterns_are_half_open() {
    let src = "fn g(n) = match n { 0..60 => \"F\", 60..70 => \"D\", 70..101 => \"P\" }\n[59, 60, 100] |> map(g)";
    assert_eq!(eval(src), Value::list(vec![s("F"), s("D"), s("P")]));
}

#[test]
fn guard_binding_then_fallthrough() {
    let src = "fn c(n) = match n { x if x < 0 => \"neg\", 0 => \"zero\", _ => \"pos\" }\n[0 - 5, 0, 5] |> map(c)";
    assert_eq!(eval(src), Value::list(vec![s("neg"), s("zero"), s("pos")]));
}

#[test]
fn question_mark_inside_an_if_expression() {
    let src =
        "fn f(x) {\n let y = if x > 0 { Some(x) } else { None }?\n Some(y * 2)\n}\n(f(5), f(0 - 1))";
    assert_eq!(
        eval(src),
        Value::tuple(vec![Value::some(Value::Int(10)), Value::none()])
    );
}

#[test]
fn empty_collection_ops_are_total() {
    assert_eq!(eval("[].first()"), Value::none());
    assert_eq!(eval("[] |> map(x => x) |> sum"), Value::Int(0));
    assert_eq!(eval("[].is_empty()"), Value::Bool(true));
}

#[test]
fn string_slice_and_len_are_char_based_not_byte_based() {
    // "héllo" is 5 chars but 6 bytes (é = 2 bytes utf-8).
    assert_eq!(eval("\"héllo\".slice(0, 3)"), s("hél"));
    assert_eq!(eval("\"héllo\".chars().len()"), Value::Int(5));
}

#[test]
fn slice_clamps_out_of_range_instead_of_faulting() {
    assert_eq!(
        eval("[1, 2, 3].slice(0, 100)"),
        Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}

#[test]
fn record_spread_overrides_and_extends() {
    // x kept from base, y overridden, z added.
    let prog = "let base = { x: 1, y: 2 }\nlet r = { ..base, y: 20, z: 3 }\n(r.x, r.y, r.z)";
    assert_eq!(
        eval(prog),
        Value::tuple(vec![Value::Int(1), Value::Int(20), Value::Int(3)])
    );
}
