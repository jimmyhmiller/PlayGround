//! The extended prelude: math, strings, lists, records, nested-path helpers
//! (get_in/assoc_in/update_in/swap_in!), and JSON.

use funct::{Funct, Value};

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

// ---------- math ----------

#[test]
fn math_functions() {
    assert_eq!(eval("sqrt(9)"), Value::Float(3.0));
    assert_eq!(eval("sqrt(2.25)"), Value::Float(1.5));
    assert_eq!(
        eval("floor(2.7) + ceil(2.1) + round(2.5)"),
        Value::Float(2.0 + 3.0 + 3.0)
    );
    assert_eq!(eval("abs(-5)"), int(5));
    assert_eq!(eval("abs(-2.5)"), Value::Float(2.5));
    assert_eq!(eval("min(3, 7)"), int(3));
    assert_eq!(eval("max(3, 7.5)"), Value::Float(7.5));
    assert_eq!(eval("clamp(99, 0, 10)"), int(10));
    assert_eq!(eval("clamp(-5, 0, 10)"), int(0));
    assert_eq!(eval("clamp(5, 0, 10)"), int(5));
    assert_eq!(eval("sin(0.0)"), Value::Float(0.0));
    assert_eq!(eval("exp(0)"), Value::Float(1.0));
    assert_eq!(eval("ln(1)"), Value::Float(0.0));
    assert_eq!(eval("log10(1000)"), Value::Float(3.0));
    assert_eq!(eval("atan2(0.0, 1.0)"), Value::Float(0.0));
}

#[test]
fn math_is_ufcs_friendly() {
    // rhai-style method calls work via UFCS
    assert_eq!(eval("(2.25).sqrt()"), Value::Float(1.5));
    assert_eq!(eval("(2.7).floor().to_int()"), int(2));
}

#[test]
fn numeric_conversions() {
    assert_eq!(eval("to_int(2.9)"), int(2)); // truncates
    assert_eq!(eval("to_int(-2.9)"), int(-2));
    assert_eq!(eval("to_int(5)"), int(5));
    assert_eq!(eval("to_float(2)"), Value::Float(2.0));
    assert_eq!(eval("parse_float(\"2.5\")"), Value::ok(Value::Float(2.5)));
    assert!(matches!(eval("parse_float(\"nope\")"), v if v == eval("Err(\"not a number: nope\")")));
}

// ---------- strings ----------

#[test]
fn string_functions() {
    assert_eq!(eval("contains(\"hello\", \"ell\")"), Value::Bool(true));
    assert_eq!(eval("starts_with(\"hello\", \"he\")"), Value::Bool(true));
    assert_eq!(eval("ends_with(\"hello\", \"lo\")"), Value::Bool(true));
    assert_eq!(eval("to_upper(\"abc\") + to_lower(\"DEF\")"), s("ABCdef"));
    assert_eq!(eval("trim(\"  x  \")"), s("x"));
    assert_eq!(eval("replace(\"a-b-c\", \"-\", \"+\")"), s("a+b+c"));
    assert_eq!(
        eval("split(\"a,b,c\", \",\")"),
        eval("[\"a\", \"b\", \"c\"]")
    );
    assert_eq!(eval("chars(\"héy\")"), eval("[\"h\", \"é\", \"y\"]"));
    assert_eq!(eval("join([\"a\", \"b\"], \"-\")"), s("a-b"));
    assert_eq!(eval("slice(\"hello\", 1, 3)"), s("ell"));
    // index_of returns Option, not -1
    assert_eq!(eval("index_of(\"hello\", \"ll\")"), Value::some(int(2)));
    assert_eq!(eval("index_of(\"hello\", \"zz\")"), Value::none());
}

#[test]
fn string_methods_via_ufcs() {
    assert_eq!(
        eval("\"a,b,c\".split(\",\") |> map(s => s.to_upper()) |> join(_, \"+\")"),
        s("A+B+C")
    );
}

#[test]
fn join_rejects_non_strings_loudly() {
    let e = eval_err("join([1, 2], \"-\")");
    assert!(e.contains("must contain only Str"), "{}", e);
}

// ---------- lists ----------

#[test]
fn list_functions() {
    assert_eq!(eval("first([1, 2, 3])"), Value::some(int(1)));
    assert_eq!(eval("first([])"), Value::none());
    assert_eq!(eval("last([1, 2, 3])"), Value::some(int(3)));
    assert_eq!(eval("rest([1, 2, 3])"), eval("[2, 3]"));
    assert_eq!(eval("rest([])"), eval("[]"));
    assert_eq!(eval("pop([1, 2, 3])"), eval("[1, 2]"));
    assert_eq!(eval("insert_at([1, 3], 1, 2)"), eval("[1, 2, 3]"));
    assert_eq!(eval("remove_at([1, 2, 3], 1)"), eval("[1, 3]"));
    assert_eq!(eval("contains([1, 2], 2)"), Value::Bool(true));
    assert_eq!(eval("contains([1, 2], 5)"), Value::Bool(false));
    assert_eq!(eval("index_of([\"a\", \"b\"], \"b\")"), Value::some(int(1)));
    assert_eq!(eval("is_empty([])"), Value::Bool(true));
    assert_eq!(eval("is_empty(\"\")"), Value::Bool(true));
    assert_eq!(eval("slice([1, 2, 3, 4], 1, 2)"), eval("[2, 3]"));
    assert_eq!(eval("sort([3, 1, 2])"), eval("[1, 2, 3]"));
    assert_eq!(eval("sort([\"b\", \"a\"])"), eval("[\"a\", \"b\"]"));
    assert_eq!(
        eval("sort_by([{ n: 2 }, { n: 1 }], r => r.n)"),
        eval("[{ n: 1 }, { n: 2 }]")
    );
}

#[test]
fn pop_empty_faults_loudly() {
    let e = eval_err("pop([])");
    assert!(e.contains("empty"), "{}", e);
}

#[test]
fn sort_mixed_types_faults_loudly() {
    let e = eval_err("sort([1, \"a\"])");
    assert!(e.contains("cannot order"), "{}", e);
}

// ---------- records ----------

#[test]
fn record_functions() {
    assert_eq!(eval("has({ x: 1 }, \"x\")"), Value::Bool(true));
    assert_eq!(eval("has({ x: 1 }, \"y\")"), Value::Bool(false));
    assert_eq!(eval("get({ x: 1 }, \"x\")"), Value::some(int(1)));
    assert_eq!(eval("get({ x: 1 }, \"y\")"), Value::none());
    assert_eq!(eval("get([10, 20], 1)"), Value::some(int(20)));
    assert_eq!(eval("get([10, 20], 9)"), Value::none());
    assert_eq!(eval("assoc({ x: 1 }, \"y\", 2)"), eval("{ x: 1, y: 2 }"));
    assert_eq!(eval("assoc([1, 2], 0, 9)"), eval("[9, 2]"));
    assert_eq!(eval("dissoc({ x: 1, y: 2 }, \"y\")"), eval("{ x: 1 }"));
    assert_eq!(
        eval("merge({ x: 1, y: 2 }, { y: 9, z: 3 })"),
        eval("{ x: 1, y: 9, z: 3 }")
    );
    assert_eq!(eval("values({ a: 1, b: 2 })"), eval("[1, 2]"));
    assert_eq!(eval("entries({ a: 1 })"), eval("[(\"a\", 1)]"));
}

// ---------- nested paths ----------

#[test]
fn get_in_and_assoc_in() {
    assert_eq!(
        eval("get_in({ a: { b: [10, 20] } }, [\"a\", \"b\", 1])"),
        Value::some(int(20))
    );
    assert_eq!(eval("get_in({ a: 1 }, [\"a\", \"b\"])"), Value::none());
    assert_eq!(
        eval("assoc_in({ a: { b: 1 } }, [\"a\", \"b\"], 9)"),
        eval("{ a: { b: 9 } }")
    );
    // missing intermediate record keys are created
    assert_eq!(
        eval("assoc_in({}, [\"a\", \"b\"], 1)"),
        eval("{ a: { b: 1 } }")
    );
    assert_eq!(
        eval("assoc_in({ xs: [1, 2] }, [\"xs\", 0], 9)"),
        eval("{ xs: [9, 2] }")
    );
}

#[test]
fn update_and_update_in() {
    assert_eq!(
        eval("update({ n: 1 }, \"n\", x => x + 1)"),
        eval("{ n: 2 }")
    );
    assert_eq!(
        eval("update_in({ a: { n: 1 } }, [\"a\", \"n\"], x => x * 10)"),
        eval("{ a: { n: 10 } }")
    );
    let e = eval_err("update({ n: 1 }, \"missing\", x => x)");
    assert!(e.contains("no value at key"), "{}", e);
}

#[test]
fn swap_in_and_reset_in() {
    let src = r#"
let state = atom({ score: 0, ui: { clicks: 0 } })
swap_in!(state, ["ui", "clicks"], n => n + 1)
swap_in!(state, ["ui", "clicks"], n => n + 1)
reset_in!(state, ["score"], 50)
(@state).ui.clicks + (@state).score
"#;
    assert_eq!(eval(src), int(52));
}

#[test]
fn swap_in_fires_watchers() {
    let src = r#"
let state = atom({ n: 0 })
let seen = atom(0)
watch(state, "w", (old, new) => swap!(seen, c => c + 1))
swap_in!(state, ["n"], x => x + 1)
reset_in!(state, ["n"], 5)
@seen
"#;
    assert_eq!(eval(src), int(2));
}

#[test]
fn swap_in_missing_path_faults_loudly() {
    let e = eval_err("let a = atom({})\nswap_in!(a, [\"missing\"], x => x)");
    assert!(e.contains("reset_in!"), "should suggest reset_in!: {}", e);
}

// ---------- json ----------

#[test]
fn json_parse_and_stringify() {
    // NB: `{`/`}` inside funct strings are interpolation, so literal JSON
    // braces in source are written \{ \}
    assert_eq!(
        eval(r#"json_parse("\{\"a\": 1, \"b\": [true, null, 2.5]\}")"#),
        Value::ok(eval("{ a: 1, b: [true, (), 2.5] }"))
    );
    assert_eq!(
        eval(r#"unwrap_or(json_stringify({ a: 1 }), "")"#),
        s(r#"{"a":1}"#)
    );
    // round trip
    assert_eq!(
        eval(
            r#"
let data = { xs: [1, 2], name: "hi", flag: true }
let encoded = unwrap_or(json_stringify(data), "")
unwrap_or(json_parse(encoded), ()) == data
"#
        ),
        Value::Bool(true)
    );
    // bad json is an Err, not a fault
    assert!(matches!(eval(r#"json_parse("\{nope")"#), v if format!("{}", v).starts_with("Err(")));
    // unrepresentable values are an Err, not a silent drop
    assert!(matches!(eval("json_stringify(atom(1))"), v if format!("{}", v).starts_with("Err(")));
}
