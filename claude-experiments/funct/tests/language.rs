//! Core language semantics, straight from funct-spec.md.

use funct::{Funct, FunctError, Value};

fn eval(src: &str) -> Value {
    let mut vm = Funct::new();
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed: {}\nsource:\n{}", e, src))
}

fn eval_err(src: &str) -> String {
    let mut vm = Funct::new();
    match vm.eval(src) {
        Ok(v) => panic!("expected error, got {:?}\nsource:\n{}", v, src),
        Err(e) => e.to_string(),
    }
}

fn int(i: i64) -> Value {
    Value::Int(i)
}

fn s(x: &str) -> Value {
    Value::str(x)
}

// ---------- literals & arithmetic ----------

#[test]
fn arithmetic_precedence() {
    assert_eq!(eval("2 + 3 * 4"), int(14));
    assert_eq!(eval("(2 + 3) * 4"), int(20));
    assert_eq!(eval("10 - 2 - 3"), int(5)); // left assoc
    assert_eq!(eval("2 ** 3 ** 2"), int(512)); // right assoc
    assert_eq!(eval("7 / 2"), int(3));
    assert_eq!(eval("7 % 3"), int(1));
    assert_eq!(eval("-2 ** 2"), int(4)); // unary binds tighter than **
}

#[test]
fn floats() {
    assert_eq!(eval("1.5 + 2.5"), Value::Float(4.0));
    assert_eq!(eval("1 + 0.5"), Value::Float(1.5));
    assert_eq!(eval("1.0 == 1"), Value::Bool(true));
}

#[test]
fn numeric_separators() {
    assert_eq!(eval("1_000_000"), int(1_000_000));
}

#[test]
fn strings_and_interpolation() {
    assert_eq!(eval(r#""hello" + " " + "world""#), s("hello world"));
    assert_eq!(eval(r#""1 + 2 = ${1 + 2}""#), s("1 + 2 = 3"));
    assert_eq!(
        eval(
            r#"let name = "ada"
"hi ${name}!""#
        ),
        s("hi ada!")
    );
    // bare braces are literal now (no escaping needed)
    assert_eq!(eval("\"a brace { and } here\""), s("a brace { and } here"));
    // `\$` escapes an interpolation
    assert_eq!(eval(r#""esc \${not interp}""#), s("esc ${not interp}"));
    assert_eq!(eval(r#""a\nb""#), s("a\nb"));
}

#[test]
fn booleans_and_short_circuit() {
    assert_eq!(eval("true and false"), Value::Bool(false));
    assert_eq!(eval("true or false"), Value::Bool(true));
    assert_eq!(eval("not true"), Value::Bool(false));
    // rhs must not evaluate: would fault with division by zero
    assert_eq!(eval("false and (1 / 0 == 0)"), Value::Bool(false));
    assert_eq!(eval("true or (1 / 0 == 0)"), Value::Bool(true));
}

#[test]
fn comparisons() {
    assert_eq!(eval("1 < 2"), Value::Bool(true));
    assert_eq!(eval("2 <= 2"), Value::Bool(true));
    assert_eq!(eval(r#""abc" < "abd""#), Value::Bool(true));
    assert_eq!(eval("1.5 > 1"), Value::Bool(true));
}

#[test]
fn structural_equality() {
    assert_eq!(eval("[1, 2] == [1, 2]"), Value::Bool(true));
    assert_eq!(eval("{ x: 1 } == { x: 1 }"), Value::Bool(true));
    assert_eq!(eval("(1, 2) == (1, 2)"), Value::Bool(true));
    assert_eq!(eval("Some(1) == Some(1)"), Value::Bool(true));
    assert_eq!(eval("Some(1) == Some(2)"), Value::Bool(false));
    // atoms compare by identity
    assert_eq!(eval("atom(1) == atom(1)"), Value::Bool(false));
    assert_eq!(eval("let a = atom(1)\na == a"), Value::Bool(true));
}

// ---------- bindings & blocks ----------

#[test]
fn let_and_shadowing() {
    assert_eq!(eval("let x = 1\nlet x = x + 1\nx"), int(2));
}

#[test]
fn blocks_are_expressions() {
    assert_eq!(eval("let y = {\n let a = 2\n a * 3\n}\ny"), int(6));
    // block without trailing expr is ()
    assert_eq!(eval("{ let a = 1\n}"), Value::Unit);
}

#[test]
fn no_top_level_mut() {
    let e = eval_err("let mut x = 1");
    assert!(e.contains("atom"), "error should point at atoms: {}", e);
}

#[test]
fn no_top_level_assignment() {
    let e = eval_err("let x = 1\nx = 2");
    assert!(e.contains("top level") || e.contains("immutable"), "{}", e);
}

#[test]
fn local_mut() {
    assert_eq!(
        eval("fn f() {\n let mut x = 1\n x = x + 1\n x += 3\n x\n}\nf()"),
        int(5)
    );
}

#[test]
fn assign_to_immutable_local_fails() {
    let e = eval_err("fn f() {\n let x = 1\n x = 2\n x\n}\nf()");
    assert!(e.contains("immutable"), "{}", e);
}

#[test]
fn destructuring_let() {
    assert_eq!(eval("let (a, b) = (1, 2)\na + b"), int(3));
    assert_eq!(eval("let [x, y, z] = [1, 2, 3]\nx + y + z"), int(6));
    assert_eq!(eval("let { x, y } = { x: 1, y: 2 }\nx + y"), int(3));
    assert_eq!(
        eval("fn f() {\n let (a, b) = (4, 5)\n a * b\n}\nf()"),
        int(20)
    );
}

// ---------- functions & closures ----------

#[test]
fn fn_definitions() {
    assert_eq!(eval("fn double(x) = x * 2\ndouble(21)"), int(42));
    assert_eq!(eval("fn add(a, b) {\n a + b\n}\nadd(1, 2)"), int(3));
}

#[test]
fn forward_references() {
    assert_eq!(eval("fn a() = b() + 1\nfn b() = 41\na()"), int(42));
}

#[test]
fn recursion() {
    assert_eq!(
        eval("fn fib(n) = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\nfib(10)"),
        int(55)
    );
}

#[test]
fn lambdas() {
    assert_eq!(eval("let f = x => x + 1\nf(41)"), int(42));
    assert_eq!(eval("let f = (x, y) => x * y\nf(6, 7)"), int(42));
    assert_eq!(eval("let f = () => 42\nf()"), int(42));
    // lambda with block body
    assert_eq!(
        eval("let f = x => {\n let y = x * 2\n y + 1\n}\nf(20)"),
        int(41)
    );
}

#[test]
fn closures_capture_immutable() {
    assert_eq!(
        eval("fn adder(n) = x => x + n\nlet add5 = adder(5)\nadd5(37)"),
        int(42)
    );
}

#[test]
fn closures_share_mutable_slot() {
    // captured `let mut` shares the slot; closures see updates (spec §4.3)
    assert_eq!(
        eval(
            "fn make_counter() {\n let mut n = 0\n () => {\n  n = n + 1\n  n\n }\n}\nlet c = make_counter()\nc()\nc()\nc()"
        ),
        int(3)
    );
}

#[test]
fn two_closures_same_slot() {
    assert_eq!(
        eval(
            r#"
fn make() {
    let mut n = 0
    let inc = () => { n = n + 1; n }
    let get = () => n
    (inc, get)
}
let (inc, get) = make()
inc()
inc()
get()
"#
        ),
        int(2)
    );
}

#[test]
fn deep_tail_recursion_is_bounded() {
    assert_eq!(
        eval("fn go(n, acc) = if n == 0 { acc } else { go(n - 1, acc + 1) }\ngo(1000000, 0)"),
        int(1_000_000)
    );
}

#[test]
fn deep_non_tail_recursion_uses_heap_frames() {
    // would blow the host stack if frames were host-recursive
    assert_eq!(
        eval("fn sum_to(n) = if n == 0 { 0 } else { n + sum_to(n - 1) }\nsum_to(50000)"),
        int(50000 * 50001 / 2)
    );
}

#[test]
fn pattern_params() {
    assert_eq!(eval("fn first((a, _)) = a\nfirst((7, 8))"), int(7));
    assert_eq!(
        eval("fn norm({ x, y }) = x * x + y * y\nnorm({ x: 3, y: 4 })"),
        int(25)
    );
}

#[test]
fn return_statement() {
    assert_eq!(
        eval("fn f(x) {\n if x > 10 {\n  return 100\n }\n x\n}\nf(50) + f(1)"),
        int(101)
    );
}

// ---------- pipes & UFCS ----------

#[test]
fn pipe_basics() {
    assert_eq!(eval("fn double(x) = x * 2\n5 |> double"), int(10));
    assert_eq!(eval("fn add(a, b) = a + b\n5 |> add(3)"), int(8));
}

#[test]
fn pipe_hole() {
    assert_eq!(eval("fn sub(a, b) = a - b\n3 |> sub(10, _)"), int(7));
    assert_eq!(eval("fn sub(a, b) = a - b\n10 |> sub(_, 3)"), int(7));
}

#[test]
fn pipe_chain_multiline() {
    assert_eq!(
        eval("[1, 2, 3, 4]\n  |> map(x => x * 2)\n  |> filter(x => x > 2)\n  |> sum"),
        int(18)
    );
}

#[test]
fn pipe_into_variant_ctor() {
    assert_eq!(eval("3 |> Some"), Value::some(int(3)));
}

#[test]
fn ufcs_method_call() {
    assert_eq!(eval("fn double(x) = x * 2\n5.double()"), int(10));
    assert_eq!(eval("fn add(a, b) = a + b\n5.add(3)"), int(8));
    assert_eq!(eval("[1, 2, 3].len()"), int(3));
}

#[test]
fn record_field_wins_over_ufcs() {
    // spec §4.1: field access wins when `f` is a field of the record
    assert_eq!(eval("let r = { double: x => x * 3 }\nr.double(5)"), int(15));
}

#[test]
fn field_access() {
    assert_eq!(eval("let p = { x: 1, y: 2 }\np.x + p.y"), int(3));
    // shorthand { x } means { x: x }
    assert_eq!(eval("let x = 9\nlet r = { x }\nr.x"), int(9));
}

#[test]
fn record_spread_update() {
    assert_eq!(
        eval("let p = { x: 1, y: 2 }\nlet p2 = { ..p, x: 10 }\np2.x + p2.y"),
        int(12)
    );
    // original untouched (immutability)
    assert_eq!(
        eval("let p = { x: 1 }\nlet p2 = { ..p, x: 10 }\np.x"),
        int(1)
    );
}

// ---------- data ----------

#[test]
fn lists() {
    assert_eq!(eval("[1, 2, 3][1]"), int(2));
    assert_eq!(eval("len([1, 2, 3])"), int(3));
    assert_eq!(eval("[1, 2] + [3]"), eval("[1, 2, 3]"));
    assert_eq!(eval("push([1], 2)"), eval("[1, 2]"));
}

#[test]
fn tuples_and_indexing() {
    assert_eq!(eval("(1, \"a\", true)[2]"), Value::Bool(true));
}

#[test]
fn string_indexing() {
    assert_eq!(eval(r#""hello"[1]"#), s("e"));
    assert_eq!(eval(r#"len("héllo")"#), int(5)); // chars, not bytes
}

#[test]
fn index_out_of_bounds_faults() {
    let e = eval_err("[1, 2][5]");
    assert!(e.contains("out of bounds"), "{}", e);
}

#[test]
fn ranges() {
    assert_eq!(eval("to_list(1..4)"), eval("[1, 2, 3]"));
    assert_eq!(eval("to_list(1..=4)"), eval("[1, 2, 3, 4]"));
}

// ---------- control flow ----------

#[test]
fn if_else() {
    assert_eq!(eval("if 1 < 2 { \"yes\" } else { \"no\" }"), s("yes"));
    assert_eq!(eval("if false { 1 }"), Value::Unit);
    assert_eq!(
        eval(
            "fn grade(n) = if n > 89 { \"A\" } else if n > 79 { \"B\" } else { \"C\" }\ngrade(85)"
        ),
        s("B")
    );
}

#[test]
fn condition_must_be_bool() {
    let e = eval_err("if 1 { 2 }");
    assert!(e.contains("Bool"), "{}", e);
}

#[test]
fn while_loop() {
    assert_eq!(
        eval("fn f() {\n let mut i = 0\n let mut total = 0\n while i < 5 {\n  i = i + 1\n  total = total + i\n }\n total\n}\nf()"),
        int(15)
    );
}

#[test]
fn for_loop_over_list_and_range() {
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for x in [1, 2, 3] { t = t + x }\n t\n}\nf()"),
        int(6)
    );
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for x in 1..=10 { t = t + x }\n t\n}\nf()"),
        int(55)
    );
}

#[test]
fn for_loop_pattern() {
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for (a, b) in [(1, 2), (3, 4)] { t = t + a * b }\n t\n}\nf()"),
        int(14)
    );
}

#[test]
fn for_loop_over_string() {
    assert_eq!(eval("fn f() {\n let mut out = \"\"\n for c in \"abc\" { out = out + c + \"-\" }\n out\n}\nf()"), s("a-b-c-"));
}

// ---------- pattern matching ----------

#[test]
fn match_literals_and_wildcard() {
    assert_eq!(
        eval("match 2 {\n 1 => \"one\",\n 2 => \"two\",\n _ => \"many\"\n}"),
        s("two")
    );
}

#[test]
fn match_variants() {
    let src = r#"
type Shape = Circle { radius: Float } | Square { side: Float } | Point
fn area(s) = match s {
    Circle { radius } => 3.14 * radius * radius,
    Square { side } => side * side,
    Point => 0.0,
}
area(Square { side: 3.0 })
"#;
    assert_eq!(eval(src), Value::Float(9.0));
}

#[test]
fn match_option_result() {
    assert_eq!(
        eval("match Some(41) {\n Some(x) => x + 1,\n None => 0\n}"),
        int(42)
    );
    assert_eq!(
        eval("match Err(\"boom\") {\n Ok(v) => v,\n Err(m) => m\n}"),
        s("boom")
    );
}

#[test]
fn match_guards() {
    assert_eq!(
        eval("fn sign(n) = match n {\n x if x > 0 => 1,\n x if x < 0 => -1,\n _ => 0\n}\nsign(-5) + sign(9) * 10"),
        int(9)
    );
}

#[test]
fn match_list_patterns() {
    assert_eq!(
        eval(
            "match [1, 2, 3, 4] {\n [] => 0,\n [x] => x,\n [first, ..rest] => first + len(rest)\n}"
        ),
        int(4)
    );
    assert_eq!(
        eval("match [] {\n [] => \"empty\",\n _ => \"no\"\n}"),
        s("empty")
    );
    // exact length
    assert_eq!(
        eval("match [1, 2] {\n [a, b, c] => 3,\n [a, b] => 2,\n _ => 0\n}"),
        int(2)
    );
}

#[test]
fn match_or_patterns() {
    assert_eq!(
        eval("fn f(n) = match n {\n 1 | 2 | 3 => \"low\",\n _ => \"high\"\n}\nf(2) + f(9)"),
        s("lowhigh")
    );
}

#[test]
fn match_range_patterns() {
    assert_eq!(
        eval("fn f(n) = match n {\n 0..10 => \"digit\",\n 10..=99 => \"two\",\n _ => \"big\"\n}\nf(5) + f(99) + f(100)"),
        s("digittwobig")
    );
}

#[test]
fn match_as_binding() {
    assert_eq!(
        eval("match [1, 2] {\n [a, _] as whole => a + len(whole)\n}"),
        int(3)
    );
}

#[test]
fn match_nested_patterns() {
    assert_eq!(
        eval("match Some((1, [2, 3])) {\n Some((a, [b, c])) => a + b + c,\n _ => 0\n}"),
        int(6)
    );
}

#[test]
fn match_tuple_and_record_patterns() {
    assert_eq!(
        // `match {` always starts a subjectless match, so a record literal
        // subject needs parens (same restriction as Rust struct literals)
        eval("match ({ x: 1, y: 2 }) {\n { x: 0, y } => y,\n { x, .. } => x * 100\n}"),
        int(100)
    );
}

#[test]
fn subjectless_match_is_a_function() {
    // spec §4.2
    assert_eq!(
        eval("let classify = match {\n 0 => \"zero\",\n n if n > 0 => \"pos\",\n _ => \"neg\"\n}\nclassify(5)"),
        s("pos")
    );
    assert_eq!(
        eval("7 |> match {\n 0 => \"zero\",\n _ => \"other\"\n}"),
        s("other")
    );
}

#[test]
fn no_match_faults() {
    let e = eval_err("match 5 {\n 1 => 1\n}");
    assert!(e.contains("no pattern matched"), "{}", e);
}

// ---------- errors: Result / Option / ? ----------

#[test]
fn question_mark_ok_path() {
    let src = r#"
fn safe_div(a, b) = if b == 0 { Err("div by zero") } else { Ok(a / b) }
fn calc(a, b) {
    let x = safe_div(a, b)?
    Ok(x + 1)
}
calc(10, 2)
"#;
    assert_eq!(eval(src), Value::ok(int(6)));
}

#[test]
fn question_mark_err_short_circuits() {
    let src = r#"
fn safe_div(a, b) = if b == 0 { Err("div by zero") } else { Ok(a / b) }
fn calc(a, b) {
    let x = safe_div(a, b)?
    Ok(x + 1)
}
calc(1, 0)
"#;
    assert_eq!(eval(src), Value::err(s("div by zero")));
}

#[test]
fn question_mark_on_option() {
    let src = r#"
fn head(xs) = match xs {
    [x, ..] => Some(x),
    [] => None,
}
fn first_doubled(xs) {
    let h = head(xs)?
    Some(h * 2)
}
(first_doubled([21]), first_doubled([]))
"#;
    let v = eval(src);
    assert_eq!(v, Value::tuple(vec![Value::some(int(42)), Value::none()]));
}

#[test]
fn division_by_zero_faults() {
    let e = eval_err("1 / 0");
    assert!(e.contains("division by zero"), "{}", e);
}

#[test]
fn unknown_variable_is_compile_error() {
    let mut vm = Funct::new();
    match vm.eval("nope + 1") {
        Err(FunctError::Compile(m)) => assert!(m.contains("nope"), "{}", m),
        other => panic!(
            "expected compile error, got {:?}",
            other.map(|v| format!("{:?}", v))
        ),
    }
}

// ---------- atoms ----------

#[test]
fn atom_basics() {
    assert_eq!(eval("let a = atom(41)\n@a + 1"), int(42));
    assert_eq!(eval("let a = atom(0)\nswap!(a, x => x + 5)\n@a"), int(5));
    assert_eq!(eval("let a = atom(0)\nreset!(a, 9)\n@a"), int(9));
    assert_eq!(eval("let a = atom(7)\nderef(a)"), int(7));
    assert_eq!(eval("let a = atom(7)\na.value"), int(7));
}

#[test]
fn atom_swap_returns_new_value() {
    assert_eq!(eval("let a = atom(1)\nswap!(a, x => x * 2)"), int(2));
}

#[test]
fn atom_pipes() {
    // spec §4.4: a |> swap!(_, f)
    assert_eq!(
        eval("let a = atom(10)\na |> swap!(_, x => x + 1)\n@a"),
        int(11)
    );
}

#[test]
fn atom_watchers() {
    let src = r#"
let a = atom(0)
let log = atom([])
watch(a, "w", (old, new) => swap!(log, xs => push(xs, (old, new))))
swap!(a, x => x + 1)
reset!(a, 10)
@log
"#;
    assert_eq!(eval(src), eval("[(0, 1), (1, 10)]"));
}

#[test]
fn atom_unwatch() {
    let src = r#"
let a = atom(0)
let log = atom(0)
watch(a, "w", (old, new) => swap!(log, n => n + 1))
swap!(a, x => x + 1)
unwatch(a, "w")
swap!(a, x => x + 1)
@log
"#;
    assert_eq!(eval(src), int(1));
}

#[test]
fn atoms_shared_through_closures() {
    let src = r#"
let counter = atom(0)
fn bump() = swap!(counter, n => n + 1)
bump()
bump()
bump()
@counter
"#;
    assert_eq!(eval(src), int(3));
}

// ---------- prelude ----------

#[test]
fn prelude_functions() {
    assert_eq!(eval("map([1, 2, 3], x => x * x)"), eval("[1, 4, 9]"));
    assert_eq!(
        eval("filter([1, 2, 3, 4], x => x % 2 == 0)"),
        eval("[2, 4]")
    );
    assert_eq!(eval("fold([1, 2, 3], 10, (a, b) => a + b)"), int(16));
    assert_eq!(eval("sum([1, 2, 3])"), int(6));
    assert_eq!(eval("reverse([1, 2, 3])"), eval("[3, 2, 1]"));
    assert_eq!(eval("unwrap_or(Some(1), 9)"), int(1));
    assert_eq!(eval("unwrap_or(None, 9)"), int(9));
    assert_eq!(eval("unwrap_or(Ok(1), 9)"), int(1));
    assert_eq!(eval("unwrap_or(Err(\"x\"), 9)"), int(9));
    assert_eq!(eval("parse_int(\"42\")"), Value::ok(int(42)));
    assert_eq!(eval("typeof(1.5)"), s("Float"));
    assert_eq!(eval("str(42) + str(true)"), s("42true"));
    assert_eq!(eval("keys({ b: 1, a: 2 })"), eval("[\"a\", \"b\"]"));
}

#[test]
fn map_over_range() {
    assert_eq!(eval("1..=3 |> map(x => x * 10)"), eval("[10, 20, 30]"));
}

#[test]
fn comments() {
    assert_eq!(eval("// a comment\n1 + 1 // trailing\n// done"), int(2));
}

#[test]
fn semicolons_separate_statements() {
    assert_eq!(eval("fn f() { let a = 1; let b = 2; a + b }\nf()"), int(3));
}

// ---------- escape analysis (let mut → cell only when captured) ----------

#[test]
fn noncaptured_mut_still_assigns() {
    // plain-slot mutable (no closure captures it): loop accumulation works
    assert_eq!(
        eval("fn f() {\n let mut t = 0\n for i in 1..=4 { t = t + i }\n t\n}\nf()"),
        int(10)
    );
}

#[test]
fn captured_and_mutated_through_closure() {
    // mutable captured AND written via a closure, driven by a loop
    assert_eq!(
        eval("fn f() {\n let mut acc = 0\n let add = x => { acc = acc + x; acc }\n for i in 1..=4 { add(i) }\n acc\n}\nf()"),
        int(10)
    );
}

#[test]
fn shadowing_outer_mut_not_captured() {
    // the lambda's own param `x` shadows the outer mut, so the outer stays a
    // plain slot and is independently assignable
    assert_eq!(
        eval("fn g() {\n let mut x = 100\n let h = x => x + 1\n x = x + 5\n (x, h(7))\n}\ng()"),
        Value::tuple(vec![int(105), int(8)])
    );
}

#[test]
fn capture_through_nested_lambda() {
    // a deeper closure captures a function-level mut through an intermediate
    // lambda — must still resolve to a shared cell
    assert_eq!(
        eval("fn outer() {\n let mut total = 0\n let mid = () => {\n  let bump = () => { total = total + 10 }\n  bump(); bump()\n }\n mid()\n total\n}\nouter()"),
        int(20)
    );
}
