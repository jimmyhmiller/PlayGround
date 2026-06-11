//! Prelude: built-in natives (atoms, I/O, basics) plus a small stdlib
//! written in funct itself (map/filter/fold/...), compiled at engine startup.

use crate::value::{Value, VariantPayload};
use crate::vm::{Fault, Funct};

pub const PRELUDE_SRC: &str = r#"
fn fold(xs, init, f) {
    let mut acc = init
    for x in xs { acc = f(acc, x) }
    acc
}

fn map(xs, f) = fold(xs, [], (acc, x) => push(acc, f(x)))

fn filter(xs, pred) = fold(xs, [], (acc, x) => if pred(x) { push(acc, x) } else { acc })

fn sum(xs) = fold(xs, 0, (a, b) => a + b)

fn to_list(xs) = fold(xs, [], (acc, x) => push(acc, x))

fn reverse(xs) {
    let mut out = []
    let mut i = len(xs)
    while i > 0 {
        i = i - 1
        out = push(out, xs[i])
    }
    out
}

fn unwrap_or(opt, default) = match opt {
    Some(x) => x,
    Ok(x) => x,
    None => default,
    Err(_) => default,
}
"#;

pub fn install(vm: &mut Funct) {
    // ----- basics -----
    vm.register_raw("str", |_vm, args| {
        expect_arity("str", &args, 1)?;
        Ok(Value::str(format!("{}", args[0])))
    });
    vm.register_raw("typeof", |_vm, args| {
        expect_arity("typeof", &args, 1)?;
        Ok(Value::str(args[0].type_name()))
    });
    vm.register_raw("print", |_vm, args| {
        let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
        print!("{}", s.join(" "));
        Ok(Value::Unit)
    });
    vm.register_raw("println", |_vm, args| {
        let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
        println!("{}", s.join(" "));
        Ok(Value::Unit)
    });
    vm.register_raw("len", |_vm, args| {
        expect_arity("len", &args, 1)?;
        let n = match &args[0] {
            Value::List(items) | Value::Tuple(items) => items.len(),
            Value::Str(s) => s.chars().count(),
            Value::Record(r) => r.len(),
            other => return Err(Fault::new(format!("len: not sized: {}", other.type_name()))),
        };
        Ok(Value::Int(n as i64))
    });
    vm.register_raw("push", |_vm, args| {
        expect_arity("push", &args, 2)?;
        match &args[0] {
            Value::List(items) => {
                let mut out = (**items).clone();
                out.push_back(args[1].clone());
                Ok(Value::list_v(out))
            }
            other => Err(Fault::new(format!("push: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("keys", |_vm, args| {
        expect_arity("keys", &args, 1)?;
        match &args[0] {
            Value::Record(r) => Ok(Value::list(r.keys().map(|k| Value::str(k.clone())).collect())),
            other => Err(Fault::new(format!("keys: expected Record, got {}", other.type_name()))),
        }
    });
    vm.register_raw("parse_int", |_vm, args| {
        expect_arity("parse_int", &args, 1)?;
        match &args[0] {
            Value::Str(s) => match s.trim().parse::<i64>() {
                Ok(i) => Ok(Value::ok(Value::Int(i))),
                Err(_) => Ok(Value::err(Value::str(format!("not an integer: {}", s)))),
            },
            other => Err(Fault::new(format!("parse_int: expected Str, got {}", other.type_name()))),
        }
    });

    // ----- atoms (the only escaping mutable state, spec §4.4) -----
    vm.register_raw("atom", |vm, args| {
        expect_arity("atom", &args, 1)?;
        Ok(vm.make_atom(args[0].clone()))
    });
    vm.register_raw("deref", |_vm, args| {
        expect_arity("deref", &args, 1)?;
        match &args[0] {
            Value::Atom(a) => Ok(a.value.read().clone()),
            other => Err(Fault::new(format!("deref: expected Atom, got {}", other.type_name()))),
        }
    });
    vm.register_raw("swap!", |vm, args| {
        expect_arity("swap!", &args, 2)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("swap!: expected Atom, got {}", other.type_name()))),
        };
        let old = atom.value.read().clone();
        // single-threaded VM: apply-and-set is atomic w.r.t. script code
        let new = vm.call_value(&args[1], vec![old.clone()])?;
        *atom.value.write() = new.clone();
        vm.fire_watchers(&atom, old, new.clone())?;
        Ok(new)
    });
    vm.register_raw("reset!", |vm, args| {
        expect_arity("reset!", &args, 2)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("reset!: expected Atom, got {}", other.type_name()))),
        };
        let old = atom.value.read().clone();
        let new = args[1].clone();
        *atom.value.write() = new.clone();
        vm.fire_watchers(&atom, old, new.clone())?;
        Ok(new)
    });
    vm.register_raw("watch", |_vm, args| {
        expect_arity("watch", &args, 3)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("watch: expected Atom, got {}", other.type_name()))),
        };
        let key = match &args[1] {
            Value::Str(s) => s.to_string(),
            other => return Err(Fault::new(format!("watch: key must be Str, got {}", other.type_name()))),
        };
        match &args[2] {
            Value::Closure(_) | Value::NativeFn(_) => {}
            other => return Err(Fault::new(format!("watch: watcher must be a function, got {}", other.type_name()))),
        }
        let mut ws = atom.watchers.write();
        ws.retain(|(k, _)| k != &key);
        ws.push((key, args[2].clone()));
        Ok(args[0].clone())
    });
    vm.register_raw("unwatch", |_vm, args| {
        expect_arity("unwatch", &args, 2)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("unwatch: expected Atom, got {}", other.type_name()))),
        };
        let key = match &args[1] {
            Value::Str(s) => s.to_string(),
            other => return Err(Fault::new(format!("unwatch: key must be Str, got {}", other.type_name()))),
        };
        atom.watchers.write().retain(|(k, _)| k != &key);
        Ok(args[0].clone())
    });

    // ----- assertions (for funct-written tests; fault loudly on failure) -----
    vm.register_raw("assert", |_vm, args| {
        if args.is_empty() || args.len() > 2 {
            return Err(Fault::new("assert expects (cond) or (cond, msg)"));
        }
        match &args[0] {
            Value::Bool(true) => Ok(Value::Unit),
            Value::Bool(false) => {
                let msg = match args.get(1) {
                    Some(Value::Str(s)) => format!("assertion failed: {}", s),
                    Some(other) => format!("assertion failed: {}", other),
                    None => "assertion failed".to_string(),
                };
                Err(Fault::new(msg))
            }
            other => Err(Fault::new(format!("assert: condition must be Bool, got {}", other.type_name()))),
        }
    });
    vm.register_raw("assert_eq", |_vm, args| {
        if args.len() < 2 || args.len() > 3 {
            return Err(Fault::new("assert_eq expects (left, right) or (left, right, msg)"));
        }
        if args[0] == args[1] {
            return Ok(Value::Unit);
        }
        let extra = match args.get(2) {
            Some(m) => format!(" — {}", m),
            None => String::new(),
        };
        Err(Fault::new(format!(
            "assert_eq failed{}:\n  left:  {:?}\n  right: {:?}",
            extra, args[0], args[1]
        )))
    });
    vm.register_raw("assert_ne", |_vm, args| {
        if args.len() < 2 || args.len() > 3 {
            return Err(Fault::new("assert_ne expects (left, right) or (left, right, msg)"));
        }
        if args[0] != args[1] {
            return Ok(Value::Unit);
        }
        let extra = match args.get(2) {
            Some(m) => format!(" — {}", m),
            None => String::new(),
        };
        Err(Fault::new(format!(
            "assert_ne failed{}: both sides are {:?}",
            extra, args[0]
        )))
    });
    vm.register_raw("fail", |_vm, args| {
        let msg = match args.first() {
            Some(Value::Str(s)) => s.to_string(),
            Some(other) => format!("{}", other),
            None => "explicit failure".to_string(),
        };
        Err(Fault::new(format!("test failed: {}", msg)))
    });

    install_math(vm);
    install_strings(vm);
    install_collections(vm);
    install_paths(vm);
    install_json(vm);

    // ----- the funct-source part of the prelude -----
    vm.eval(PRELUDE_SRC).expect("prelude must compile and run");
}

// ---------- math ----------

fn install_math(vm: &mut Funct) {
    vm.register1("sqrt", |x: f64| x.sqrt());
    vm.register1("sin", |x: f64| x.sin());
    vm.register1("cos", |x: f64| x.cos());
    vm.register1("tan", |x: f64| x.tan());
    vm.register2("atan2", |y: f64, x: f64| y.atan2(x));
    vm.register1("exp", |x: f64| x.exp());
    vm.register1("ln", |x: f64| x.ln());
    vm.register1("log10", |x: f64| x.log10());
    vm.register1("floor", |x: f64| x.floor());
    vm.register1("ceil", |x: f64| x.ceil());
    vm.register1("round", |x: f64| x.round());
    vm.register_raw("abs", |_vm, args| {
        expect_arity("abs", &args, 1)?;
        match &args[0] {
            Value::Int(i) => Ok(Value::Int(i.checked_abs().ok_or_else(|| Fault::new("abs: integer overflow"))?)),
            Value::Float(f) => Ok(Value::Float(f.abs())),
            other => Err(Fault::new(format!("abs: expected a number, got {}", other.type_name()))),
        }
    });
    vm.register_raw("min", |_vm, args| {
        expect_arity("min", &args, 2)?;
        num_pick("min", &args[0], &args[1], |a, b| a <= b)
    });
    vm.register_raw("max", |_vm, args| {
        expect_arity("max", &args, 2)?;
        num_pick("max", &args[0], &args[1], |a, b| a >= b)
    });
    vm.register_raw("clamp", |_vm, args| {
        expect_arity("clamp", &args, 3)?;
        let lo = num_pick("clamp", &args[0], &args[1], |a, b| a >= b)?; // max(x, lo)
        num_pick("clamp", &lo, &args[2], |a, b| a <= b) // min(.., hi)
    });
    vm.register_raw("to_int", |_vm, args| {
        expect_arity("to_int", &args, 1)?;
        match &args[0] {
            Value::Int(i) => Ok(Value::Int(*i)),
            Value::Float(f) => {
                if f.is_finite() && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                    Ok(Value::Int(f.trunc() as i64))
                } else {
                    Err(Fault::new(format!("to_int: {} is out of integer range", f)))
                }
            }
            other => Err(Fault::new(format!("to_int: expected a number, got {}", other.type_name()))),
        }
    });
    vm.register_raw("to_float", |_vm, args| {
        expect_arity("to_float", &args, 1)?;
        match &args[0] {
            Value::Int(i) => Ok(Value::Float(*i as f64)),
            Value::Float(f) => Ok(Value::Float(*f)),
            other => Err(Fault::new(format!("to_float: expected a number, got {}", other.type_name()))),
        }
    });
    vm.register_raw("parse_float", |_vm, args| {
        expect_arity("parse_float", &args, 1)?;
        match &args[0] {
            Value::Str(s) => match s.trim().parse::<f64>() {
                Ok(f) => Ok(Value::ok(Value::Float(f))),
                Err(_) => Ok(Value::err(Value::str(format!("not a number: {}", s)))),
            },
            other => Err(Fault::new(format!("parse_float: expected Str, got {}", other.type_name()))),
        }
    });
}

fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

fn num_pick(name: &str, a: &Value, b: &Value, take_a: fn(f64, f64) -> bool) -> Result<Value, Fault> {
    match (as_num(a), as_num(b)) {
        (Some(x), Some(y)) => Ok(if take_a(x, y) { a.clone() } else { b.clone() }),
        _ => Err(Fault::new(format!(
            "{}: expected numbers, got {} and {}",
            name,
            a.type_name(),
            b.type_name()
        ))),
    }
}

// ---------- strings ----------

fn install_strings(vm: &mut Funct) {
    vm.register2("starts_with", |s: String, p: String| s.starts_with(&p));
    vm.register2("ends_with", |s: String, p: String| s.ends_with(&p));
    vm.register1("to_lower", |s: String| s.to_lowercase());
    vm.register1("to_upper", |s: String| s.to_uppercase());
    vm.register1("trim", |s: String| s.trim().to_string());
    vm.register3("replace", |s: String, from: String, to: String| s.replace(&from, &to));
    vm.register_raw("split", |_vm, args| {
        expect_arity("split", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(sep)) => {
                if sep.is_empty() {
                    return Err(Fault::new("split: separator must be non-empty (use chars() for characters)"));
                }
                Ok(Value::list(s.split(&**sep).map(Value::str).collect()))
            }
            (a, b) => Err(Fault::new(format!("split: expected (Str, Str), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("chars", |_vm, args| {
        expect_arity("chars", &args, 1)?;
        match &args[0] {
            Value::Str(s) => Ok(Value::list(s.chars().map(|c| Value::str(c.to_string())).collect())),
            other => Err(Fault::new(format!("chars: expected Str, got {}", other.type_name()))),
        }
    });
    vm.register_raw("join", |_vm, args| {
        expect_arity("join", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::List(items), Value::Str(sep)) => {
                let parts: Result<Vec<String>, Fault> = items
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) => Ok(s.to_string()),
                        other => Err(Fault::new(format!(
                            "join: list must contain only Str, found {} (use map(xs, str) first)",
                            other.type_name()
                        ))),
                    })
                    .collect();
                Ok(Value::str(parts?.join(&**sep)))
            }
            (a, b) => Err(Fault::new(format!("join: expected (List, Str), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
}

// ---------- lists / records ----------

fn install_collections(vm: &mut Funct) {
    vm.register_raw("contains", |_vm, args| {
        expect_arity("contains", &args, 2)?;
        match &args[0] {
            Value::Str(s) => match &args[1] {
                Value::Str(needle) => Ok(Value::Bool(s.contains(&**needle))),
                other => Err(Fault::new(format!("contains: needle for a Str must be Str, got {}", other.type_name()))),
            },
            Value::List(items) => Ok(Value::Bool(items.iter().any(|v| v == &args[1]))),
            other => Err(Fault::new(format!("contains: expected Str or List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("index_of", |_vm, args| {
        expect_arity("index_of", &args, 2)?;
        match &args[0] {
            Value::Str(s) => match &args[1] {
                Value::Str(needle) => Ok(match s.find(&**needle) {
                    // byte offset -> char index
                    Some(byte) => Value::some(Value::Int(s[..byte].chars().count() as i64)),
                    None => Value::none(),
                }),
                other => Err(Fault::new(format!("index_of: needle for a Str must be Str, got {}", other.type_name()))),
            },
            Value::List(items) => Ok(match items.iter().position(|v| v == &args[1]) {
                Some(i) => Value::some(Value::Int(i as i64)),
                None => Value::none(),
            }),
            other => Err(Fault::new(format!("index_of: expected Str or List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("is_empty", |_vm, args| {
        expect_arity("is_empty", &args, 1)?;
        let empty = match &args[0] {
            Value::Str(s) => s.is_empty(),
            Value::List(items) | Value::Tuple(items) => items.is_empty(),
            Value::Record(r) => r.is_empty(),
            other => return Err(Fault::new(format!("is_empty: not sized: {}", other.type_name()))),
        };
        Ok(Value::Bool(empty))
    });
    vm.register_raw("slice", |_vm, args| {
        expect_arity("slice", &args, 3)?;
        let (start, count) = match (&args[1], &args[2]) {
            (Value::Int(a), Value::Int(b)) if *a >= 0 && *b >= 0 => (*a as usize, *b as usize),
            (a, b) => {
                return Err(Fault::new(format!(
                    "slice: start and length must be non-negative Ints, got {} and {}",
                    a.type_name(),
                    b.type_name()
                )))
            }
        };
        match &args[0] {
            Value::Str(s) => Ok(Value::str(s.chars().skip(start).take(count).collect::<String>())),
            Value::List(items) => Ok(Value::list(
                items.iter().skip(start).take(count).cloned().collect(),
            )),
            other => Err(Fault::new(format!("slice: expected Str or List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("first", |_vm, args| {
        expect_arity("first", &args, 1)?;
        match &args[0] {
            Value::List(items) => Ok(items.front().cloned().map(Value::some).unwrap_or_else(Value::none)),
            other => Err(Fault::new(format!("first: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("last", |_vm, args| {
        expect_arity("last", &args, 1)?;
        match &args[0] {
            Value::List(items) => Ok(items.last().cloned().map(Value::some).unwrap_or_else(Value::none)),
            other => Err(Fault::new(format!("last: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("rest", |_vm, args| {
        expect_arity("rest", &args, 1)?;
        match &args[0] {
            Value::List(items) => Ok(Value::list(items.iter().skip(1).cloned().collect())),
            other => Err(Fault::new(format!("rest: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("pop", |_vm, args| {
        expect_arity("pop", &args, 1)?;
        match &args[0] {
            Value::List(items) => {
                if items.is_empty() {
                    return Err(Fault::new("pop: list is empty"));
                }
                let mut out = (**items).clone();
                out.pop_back();
                Ok(Value::list_v(out))
            }
            other => Err(Fault::new(format!("pop: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("insert_at", |_vm, args| {
        expect_arity("insert_at", &args, 3)?;
        match (&args[0], &args[1]) {
            (Value::List(items), Value::Int(i)) if *i >= 0 && (*i as usize) <= items.len() => {
                let mut out = (**items).clone();
                out.insert(*i as usize, args[2].clone());
                Ok(Value::list_v(out))
            }
            (Value::List(items), Value::Int(i)) => Err(Fault::new(format!(
                "insert_at: index {} out of bounds (length {})",
                i,
                items.len()
            ))),
            (a, b) => Err(Fault::new(format!("insert_at: expected (List, Int), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("remove_at", |_vm, args| {
        expect_arity("remove_at", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::List(items), Value::Int(i)) if *i >= 0 && (*i as usize) < items.len() => {
                let mut out = (**items).clone();
                out.remove(*i as usize);
                Ok(Value::list_v(out))
            }
            (Value::List(items), Value::Int(i)) => Err(Fault::new(format!(
                "remove_at: index {} out of bounds (length {})",
                i,
                items.len()
            ))),
            (a, b) => Err(Fault::new(format!("remove_at: expected (List, Int), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("sort", |_vm, args| {
        expect_arity("sort", &args, 1)?;
        match &args[0] {
            Value::List(items) => {
                let mut out: Vec<Value> = items.iter().cloned().collect();
                sort_values("sort", &mut out)?;
                Ok(Value::list(out))
            }
            other => Err(Fault::new(format!("sort: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("sort_by", |vm, args| {
        expect_arity("sort_by", &args, 2)?;
        match &args[0] {
            Value::List(items) => {
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(items.len());
                for v in items.iter() {
                    let key = vm.call_value(&args[1], vec![v.clone()])?;
                    keyed.push((key, v.clone()));
                }
                let mut keys: Vec<Value> = keyed.iter().map(|(k, _)| k.clone()).collect();
                sort_values("sort_by", &mut keys)?; // validates key types
                keyed.sort_by(|(a, _), (b, _)| cmp_values(a, b).unwrap_or(std::cmp::Ordering::Equal));
                Ok(Value::list(keyed.into_iter().map(|(_, v)| v).collect()))
            }
            other => Err(Fault::new(format!("sort_by: expected List, got {}", other.type_name()))),
        }
    });
    vm.register_raw("has", |_vm, args| {
        expect_arity("has", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::Record(r), Value::Str(k)) => Ok(Value::Bool(r.contains_key(&**k))),
            (a, b) => Err(Fault::new(format!("has: expected (Record, Str), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("get", |_vm, args| {
        expect_arity("get", &args, 2)?;
        Ok(get_key(&args[0], &args[1])?.map(Value::some).unwrap_or_else(Value::none))
    });
    vm.register_raw("assoc", |_vm, args| {
        expect_arity("assoc", &args, 3)?;
        assoc_key(&args[0], &args[1], args[2].clone())
    });
    vm.register_raw("dissoc", |_vm, args| {
        expect_arity("dissoc", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::Record(r), Value::Str(k)) => {
                let mut out = r.clone();
                out.remove(&**k);
                Ok(Value::Record(out))
            }
            (a, b) => Err(Fault::new(format!("dissoc: expected (Record, Str), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("merge", |_vm, args| {
        expect_arity("merge", &args, 2)?;
        match (&args[0], &args[1]) {
            (Value::Record(a), Value::Record(b)) => {
                let mut out = a.clone();
                for (k, v) in b.iter() {
                    out.insert(k.clone(), v.clone());
                }
                Ok(Value::Record(out))
            }
            (a, b) => Err(Fault::new(format!("merge: expected (Record, Record), got ({}, {})", a.type_name(), b.type_name()))),
        }
    });
    vm.register_raw("values", |_vm, args| {
        expect_arity("values", &args, 1)?;
        match &args[0] {
            Value::Record(r) => Ok(Value::list(r.values().cloned().collect())),
            other => Err(Fault::new(format!("values: expected Record, got {}", other.type_name()))),
        }
    });
    vm.register_raw("entries", |_vm, args| {
        expect_arity("entries", &args, 1)?;
        match &args[0] {
            Value::Record(r) => Ok(Value::list(
                r.iter()
                    .map(|(k, v)| Value::tuple(vec![Value::str(k.clone()), v.clone()]))
                    .collect(),
            )),
            other => Err(Fault::new(format!("entries: expected Record, got {}", other.type_name()))),
        }
    });
}

fn cmp_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        _ => match (as_num(a), as_num(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y),
            _ => None,
        },
    }
}

fn sort_values(name: &str, items: &mut [Value]) -> Result<(), Fault> {
    for w in items.windows(2) {
        if cmp_values(&w[0], &w[1]).is_none() {
            return Err(Fault::new(format!(
                "{}: cannot order {} and {} (sortable: all numbers or all strings)",
                name,
                w[0].type_name(),
                w[1].type_name()
            )));
        }
    }
    items.sort_by(|a, b| cmp_values(a, b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(())
}

// ---------- nested access (get_in / assoc_in / update / swap_in!) ----------

fn get_key(container: &Value, key: &Value) -> Result<Option<Value>, Fault> {
    match (container, key) {
        (Value::Record(r), Value::Str(k)) => Ok(r.get(&**k).cloned()),
        (Value::List(items), Value::Int(i)) | (Value::Tuple(items), Value::Int(i)) => {
            if *i >= 0 {
                Ok(items.get(*i as usize).cloned())
            } else {
                Ok(None)
            }
        }
        (c, k) => Err(Fault::new(format!(
            "cannot look up a {} key in {}",
            k.type_name(),
            c.type_name()
        ))),
    }
}

fn assoc_key(container: &Value, key: &Value, v: Value) -> Result<Value, Fault> {
    match (container, key) {
        (Value::Record(r), Value::Str(k)) => {
            let mut out = r.clone();
            out.insert(k.to_string(), v);
            Ok(Value::Record(out))
        }
        (Value::List(items), Value::Int(i)) => {
            if *i >= 0 && (*i as usize) < items.len() {
                let mut out = (**items).clone();
                out[*i as usize] = v;
                Ok(Value::list_v(out))
            } else {
                Err(Fault::new(format!("assoc: index {} out of bounds (length {})", i, items.len())))
            }
        }
        (c, k) => Err(Fault::new(format!(
            "assoc: cannot set a {} key in {}",
            k.type_name(),
            c.type_name()
        ))),
    }
}

fn path_keys(path: &Value) -> Result<Vec<Value>, Fault> {
    match path {
        Value::List(items) => Ok(items.iter().cloned().collect()),
        other => Err(Fault::new(format!("path must be a List of keys, got {}", other.type_name()))),
    }
}

fn get_in(container: &Value, keys: &[Value]) -> Result<Option<Value>, Fault> {
    let mut cur = container.clone();
    for k in keys {
        // navigating into a non-container just means "nothing there" — the
        // None result is the signal (assoc_in, by contrast, faults loudly)
        let next = match (&cur, k) {
            (Value::Record(_), Value::Str(_))
            | (Value::List(_), Value::Int(_))
            | (Value::Tuple(_), Value::Int(_)) => get_key(&cur, k)?,
            _ => None,
        };
        match next {
            Some(n) => cur = n,
            None => return Ok(None),
        }
    }
    Ok(Some(cur))
}

/// assoc_in: missing intermediate RECORD keys are created as empty records
/// (Clojure-style); anything else missing or mistyped faults loudly.
fn assoc_in(container: &Value, keys: &[Value], v: Value) -> Result<Value, Fault> {
    match keys {
        [] => Ok(v),
        [k, rest @ ..] => {
            let inner = match get_key(container, k)? {
                Some(x) => x,
                None => match (container, k) {
                    (Value::Record(_), Value::Str(_)) if !rest.is_empty() => {
                        Value::record(std::collections::BTreeMap::new())
                    }
                    (Value::Record(_), Value::Str(_)) => Value::Unit, // replaced below anyway
                    _ => {
                        return Err(Fault::new(format!(
                            "assoc_in: nothing at key {:?} in {}",
                            k,
                            container.type_name()
                        )))
                    }
                },
            };
            let new_inner = assoc_in(&inner, rest, v)?;
            assoc_key(container, k, new_inner)
        }
    }
}

fn install_paths(vm: &mut Funct) {
    vm.register_raw("get_in", |_vm, args| {
        expect_arity("get_in", &args, 2)?;
        let keys = path_keys(&args[1])?;
        Ok(get_in(&args[0], &keys)?.map(Value::some).unwrap_or_else(Value::none))
    });
    vm.register_raw("assoc_in", |_vm, args| {
        expect_arity("assoc_in", &args, 3)?;
        let keys = path_keys(&args[1])?;
        if keys.is_empty() {
            return Err(Fault::new("assoc_in: path must not be empty"));
        }
        assoc_in(&args[0], &keys, args[2].clone())
    });
    vm.register_raw("update", |vm, args| {
        expect_arity("update", &args, 3)?;
        let cur = get_key(&args[0], &args[1])?.ok_or_else(|| {
            Fault::new(format!("update: no value at key {:?} (use assoc to add one)", args[1]))
        })?;
        let new = vm.call_value(&args[2], vec![cur])?;
        assoc_key(&args[0], &args[1], new)
    });
    vm.register_raw("update_in", |vm, args| {
        expect_arity("update_in", &args, 3)?;
        let keys = path_keys(&args[1])?;
        if keys.is_empty() {
            return Err(Fault::new("update_in: path must not be empty"));
        }
        let cur = get_in(&args[0], &keys)?.ok_or_else(|| {
            Fault::new(format!("update_in: no value at path {} (use assoc_in to add one)", args[1]))
        })?;
        let new = vm.call_value(&args[2], vec![cur])?;
        assoc_in(&args[0], &keys, new)
    });
    vm.register_raw("swap_in!", |vm, args| {
        expect_arity("swap_in!", &args, 3)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("swap_in!: expected Atom, got {}", other.type_name()))),
        };
        let keys = path_keys(&args[1])?;
        let old = atom.value.read().clone();
        let cur = get_in(&old, &keys)?.ok_or_else(|| {
            Fault::new(format!("swap_in!: no value at path {} (use reset_in! to create it)", args[1]))
        })?;
        let new_leaf = vm.call_value(&args[2], vec![cur])?;
        let new = assoc_in(&old, &keys, new_leaf)?;
        *atom.value.write() = new.clone();
        vm.fire_watchers(&atom, old, new.clone())?;
        Ok(new)
    });
    vm.register_raw("reset_in!", |vm, args| {
        expect_arity("reset_in!", &args, 3)?;
        let atom = match &args[0] {
            Value::Atom(a) => a.clone(),
            other => return Err(Fault::new(format!("reset_in!: expected Atom, got {}", other.type_name()))),
        };
        let keys = path_keys(&args[1])?;
        let old = atom.value.read().clone();
        let new = assoc_in(&old, &keys, args[2].clone())?;
        *atom.value.write() = new.clone();
        vm.fire_watchers(&atom, old, new.clone())?;
        Ok(new)
    });
}

// ---------- json ----------

fn install_json(vm: &mut Funct) {
    vm.register_raw("json_parse", |_vm, args| {
        expect_arity("json_parse", &args, 1)?;
        match &args[0] {
            Value::Str(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(j) => Ok(Value::ok(Value::from_json(&j))),
                Err(e) => Ok(Value::err(Value::str(format!("invalid JSON: {}", e)))),
            },
            other => Err(Fault::new(format!("json_parse: expected Str, got {}", other.type_name()))),
        }
    });
    vm.register_raw("json_stringify", |_vm, args| {
        expect_arity("json_stringify", &args, 1)?;
        match args[0].to_json() {
            Ok(j) => Ok(Value::ok(Value::str(j.to_string()))),
            Err(e) => Ok(Value::err(Value::str(e.msg))),
        }
    });
}

fn expect_arity(name: &str, args: &[Value], n: usize) -> Result<(), Fault> {
    if args.len() != n {
        return Err(Fault::new(format!("{} expects {} argument(s), got {}", name, n, args.len())));
    }
    Ok(())
}

/// Helper for natives that want a variant payload.
#[allow(dead_code)]
pub fn variant_payload(v: &Value) -> Option<(&str, &VariantPayload)> {
    match v {
        Value::Variant(var) => Some((var.tag.as_str(), &var.payload)),
        _ => None,
    }
}
