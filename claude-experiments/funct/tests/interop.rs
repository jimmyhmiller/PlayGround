//! Rust interop (spec §9): registering functions, host types with fields &
//! methods, modules, value conversions in both directions, and calling
//! script functions from Rust.

use funct::vals;
use funct::{Fault, Funct, Value};

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn register_simple_fn() {
    let mut vm = Funct::new();
    vm.register1("double", |x: i64| x * 2);
    assert_eq!(vm.eval("double(21)").unwrap(), int(42));
    // automatically UFCS + pipeable (spec §4.1)
    assert_eq!(vm.eval("21.double()").unwrap(), int(42));
    assert_eq!(vm.eval("21 |> double").unwrap(), int(42));
}

#[test]
fn register_various_arities() {
    let mut vm = Funct::new();
    vm.register0("zero", || 0i64);
    vm.register2("addi", |a: i64, b: i64| a + b);
    vm.register3("clamp", |x: i64, lo: i64, hi: i64| x.max(lo).min(hi));
    assert_eq!(vm.eval("zero()").unwrap(), int(0));
    assert_eq!(vm.eval("addi(40, 2)").unwrap(), int(42));
    assert_eq!(vm.eval("clamp(99, 0, 10)").unwrap(), int(10));
}

#[test]
fn type_conversions() {
    let mut vm = Funct::new();
    vm.register1("sum_list", |xs: Vec<i64>| xs.iter().sum::<i64>());
    vm.register1("shout", |s: String| format!("{}!", s.to_uppercase()));
    vm.register1("halve", |x: f64| x / 2.0);
    vm.register1("maybe_inc", |x: Option<i64>| x.map(|v| v + 1));
    assert_eq!(vm.eval("sum_list([1, 2, 3])").unwrap(), int(6));
    assert_eq!(vm.eval("shout(\"hey\")").unwrap(), Value::str("HEY!"));
    assert_eq!(vm.eval("halve(5)").unwrap(), Value::Float(2.5)); // Int coerces to f64
    assert_eq!(vm.eval("maybe_inc(Some(1))").unwrap(), Value::some(int(2)));
    assert_eq!(vm.eval("maybe_inc(None)").unwrap(), Value::none());
}

#[test]
fn wrong_arg_type_faults_loudly() {
    let mut vm = Funct::new();
    vm.register1("double", |x: i64| x * 2);
    let err = vm.eval("double(\"nope\")").unwrap_err().to_string();
    assert!(err.contains("expected Int"), "{}", err);
}

#[test]
fn rust_result_becomes_script_result() {
    let mut vm = Funct::new();
    vm.register1("read_config", |key: String| -> Result<i64, String> {
        match key.as_str() {
            "port" => Ok(8080),
            other => Err(format!("no key {}", other)),
        }
    });
    assert_eq!(vm.eval("read_config(\"port\")").unwrap(), Value::ok(int(8080)));
    // works with `?` in script
    let src = r#"
fn port_plus_one() {
    let p = read_config("port")?
    Ok(p + 1)
}
port_plus_one()
"#;
    assert_eq!(vm.eval(src).unwrap(), Value::ok(int(8081)));
    let src2 = "fn f() {\n let v = read_config(\"missing\")?\n Ok(v)\n}\nf()";
    assert_eq!(vm.eval(src2).unwrap(), Value::err(Value::str("no key missing")));
}

#[test]
fn call_script_from_rust() {
    let mut vm = Funct::new();
    vm.eval("fn compute_damage(weapon, target) = weapon * 2 + target").unwrap();
    let dmg: i64 = vm.call_typed("compute_damage", vals![20, 2]).unwrap();
    assert_eq!(dmg, 42);
    // typed conversion of richer results
    vm.eval("fn pair(a, b) = (a, b + 1)").unwrap();
    let (a, b): (i64, i64) = vm.call_typed("pair", vals![1, 2]).unwrap();
    assert_eq!((a, b), (1, 3));
    // script Result -> Rust Result
    vm.eval("fn checked(n) = if n > 0 { Ok(n) } else { Err(\"neg\") }").unwrap();
    let ok: Result<i64, String> = vm.call_typed("checked", vals![5]).unwrap();
    assert_eq!(ok, Ok(5));
    let err: Result<i64, String> = vm.call_typed("checked", vals![-1]).unwrap();
    assert_eq!(err, Err("neg".to_string()));
}

#[test]
fn native_taking_script_closure() {
    let mut vm = Funct::new();
    vm.register_raw("apply_twice", |vm, args| {
        if args.len() != 2 {
            return Err(Fault::new("apply_twice expects (f, x)"));
        }
        let once = vm.call_value(&args[0], vec![args[1].clone()])?;
        vm.call_value(&args[0], vec![once])
    });
    assert_eq!(vm.eval("apply_twice(x => x * 3, 2)").unwrap(), int(18));
    // and with a script-defined named fn
    assert_eq!(vm.eval("fn inc(x) = x + 1\napply_twice(inc, 40)").unwrap(), int(42));
}

#[test]
fn host_type_fields_and_methods() {
    #[derive(Clone)]
    struct Player {
        name: String,
        hp: i64,
    }
    let mut vm = Funct::new();
    vm.register_type::<Player>("Player")
        .ctor2("new_player", |name: String, hp: i64| Player { name, hp })
        .field("name", |p| p.name.clone())
        .field("hp", |p| p.hp)
        .method1("damage", |p, n: i64| {
            p.hp -= n;
            p.hp
        })
        .method0("is_dead", |p| p.hp <= 0);

    let src = r#"
let p = new_player("hero", 10)
p.damage(3)
p |> damage(2)
damage(p, 1)
(p.name, p.hp, p.is_dead())
"#;
    let v = vm.eval(src).unwrap();
    assert_eq!(
        v,
        Value::tuple(vec![Value::str("hero"), int(4), Value::Bool(false)])
    );
}

#[test]
fn host_value_mutations_visible_from_rust() {
    struct Counter {
        n: i64,
    }
    let mut vm = Funct::new();
    vm.register_type::<Counter>("Counter")
        .field("n", |c| c.n)
        .method0("incr", |c| {
            c.n += 1;
            c.n
        });
    // host creates the value, hands it to the script, reads it back after
    let counter = Value::native("Counter", Counter { n: 0 });
    let g = {
        // stash it as a global by registering a tiny accessor
        let c2 = counter.clone();
        vm.register_raw("the_counter", move |_vm, _args| Ok(c2.clone()));
        "the_counter()"
    };
    vm.eval(&format!("let c = {}\nc.incr()\nc.incr()\nc.incr()", g)).unwrap();
    match &counter {
        Value::Native(n) => {
            let count = n.with_ref::<Counter, _>(|c| c.n).unwrap();
            assert_eq!(count, 3);
        }
        _ => unreachable!(),
    }
}

#[test]
fn unknown_field_on_host_type_faults() {
    struct Thing;
    let mut vm = Funct::new();
    vm.register_type::<Thing>("Thing").ctor0("thing", || Thing);
    let err = vm.eval("thing().bogus").unwrap_err().to_string();
    assert!(err.contains("no registered field `bogus`"), "{}", err);
}

#[test]
fn register_module() {
    let mut vm = Funct::new();
    vm.register3("lerp_impl", |a: f64, b: f64, t: f64| a + (b - a) * t);
    vm.register1("abs_impl", |x: f64| x.abs());
    let lerp = vm.native_fn("lerp_impl").unwrap();
    let abs = vm.native_fn("abs_impl").unwrap();
    vm.register_module("math", vec![("lerp", lerp), ("abs", abs)]);
    assert_eq!(vm.eval("math.lerp(0.0, 10.0, 0.5)").unwrap(), Value::Float(5.0));
    assert_eq!(vm.eval("math.abs(0.0 - 3.0)").unwrap(), Value::Float(3.0));
}

#[test]
fn host_reads_script_atoms() {
    let mut vm = Funct::new();
    vm.eval("let state = atom({ score: 0 })").unwrap();
    vm.eval("swap!(state, s => { ..s, score: 10 })").unwrap();
    match vm.global("state").unwrap() {
        Value::Atom(a) => {
            let v = a.value.read().clone();
            match v {
                Value::Record(r) => assert_eq!(r.get("score"), Some(&int(10))),
                other => panic!("expected record, got {:?}", other),
            }
        }
        other => panic!("expected atom, got {:?}", other),
    }
}

#[test]
fn redefining_a_native_replaces_it() {
    let mut vm = Funct::new();
    vm.register1("f", |x: i64| x + 1);
    assert_eq!(vm.eval("f(1)").unwrap(), int(2));
    vm.register1("f", |x: i64| x * 10);
    assert_eq!(vm.eval("f(1)").unwrap(), int(10));
}
