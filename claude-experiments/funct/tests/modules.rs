//! Module system: explicit imports only — named `import { a, b } from "m"`
//! (with `as` renames) or qualified `import "m" as m`. There is NO wildcard
//! import, by design.

use funct::{Funct, Value};
use std::path::PathBuf;

fn int(i: i64) -> Value {
    Value::Int(i)
}

/// Create a temp module root with the given (relative path, source) files.
fn module_root(files: &[(&str, &str)]) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "funct_mod_test_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    for (path, src) in files {
        let full = dir.join(format!("{}.ft", path));
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, src).unwrap();
    }
    dir
}

fn vm_with(files: &[(&str, &str)]) -> Funct {
    let mut vm = Funct::new();
    vm.set_module_root(module_root(files));
    vm
}

const MATH_MOD: &str = r#"
export fn double(x) = x * 2
export fn triple(x) = x * 3
fn private_helper(x) = x + 1000
export let answer = 42
let secret = 7
"#;

#[test]
fn named_imports() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    assert_eq!(
        vm.eval("import { double, answer } from \"math\"\ndouble(answer)")
            .unwrap(),
        int(84)
    );
}

#[test]
fn named_import_with_alias() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    assert_eq!(
        vm.eval("import { double as twice, triple } from \"math\"\ntwice(1) + triple(1)")
            .unwrap(),
        int(5)
    );
}

#[test]
fn qualified_import_with_alias() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    assert_eq!(
        vm.eval("import \"math\" as m\nm.double(10) + m.answer")
            .unwrap(),
        int(62)
    );
}

#[test]
fn qualified_import_default_alias_is_last_segment() {
    let mut vm = vm_with(&[("util/math", MATH_MOD)]);
    assert_eq!(
        vm.eval("import \"util/math\"\nmath.double(21)").unwrap(),
        int(42)
    );
}

#[test]
fn nested_module_paths() {
    let mut vm = vm_with(&[("util/strings/case", "export fn shout(s) = s + \"!\"")]);
    assert_eq!(
        vm.eval("import { shout } from \"util/strings/case\"\nshout(\"hi\")")
            .unwrap(),
        Value::str("hi!")
    );
}

#[test]
fn no_wildcard_import_top_level_star() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let err = vm.eval("import * from \"math\"").unwrap_err().to_string();
    assert!(err.contains("wildcard"), "{}", err);
}

#[test]
fn no_wildcard_import_in_braces() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let err = vm
        .eval("import { * } from \"math\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("wildcard"), "{}", err);
}

#[test]
fn non_exported_items_are_not_importable() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let err = vm
        .eval("import { private_helper } from \"math\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no export `private_helper`"), "{}", err);
    // the error lists what IS available
    assert!(err.contains("double") && err.contains("answer"), "{}", err);
    let err2 = vm
        .eval("import { secret } from \"math\"")
        .unwrap_err()
        .to_string();
    assert!(err2.contains("no export `secret`"), "{}", err2);
}

#[test]
fn qualified_record_only_contains_exports() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let err = vm
        .eval("import \"math\" as m\nm.private_helper(1)")
        .unwrap_err()
        .to_string();
    assert!(err.contains("private_helper"), "{}", err);
}

#[test]
fn missing_module_fails_loudly() {
    let mut vm = vm_with(&[]);
    let err = vm
        .eval("import { x } from \"nope\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot load module `nope`"), "{}", err);
}

#[test]
fn module_importing_module() {
    let mut vm = vm_with(&[
        ("base", "export fn one() = 1"),
        (
            "derived",
            "import { one } from \"base\"\nexport fn two() = one() + one()",
        ),
    ]);
    assert_eq!(
        vm.eval("import { two } from \"derived\"\ntwo()").unwrap(),
        int(2)
    );
}

#[test]
fn circular_imports_fail_loudly() {
    let mut vm = vm_with(&[
        ("a", "import { b } from \"b\"\nexport fn a() = 1"),
        ("b", "import { a } from \"a\"\nexport fn b() = 2"),
    ]);
    let err = vm.eval("import { a } from \"a\"").unwrap_err().to_string();
    assert!(err.contains("circular"), "{}", err);
    assert!(err.contains("a -> b -> a"), "{}", err);
}

#[test]
fn modules_are_loaded_once_and_share_state() {
    // both importers see the SAME module instance (same atom)
    let counter_mod = r#"
let count = atom(0)
export fn bump() = swap!(count, n => n + 1)
export fn current() = @count
"#;
    let mut vm = vm_with(&[
        ("counter", counter_mod),
        (
            "user_a",
            "import { bump } from \"counter\"\nexport fn poke_a() = bump()",
        ),
        (
            "user_b",
            "import { bump } from \"counter\"\nexport fn poke_b() = bump()",
        ),
    ]);
    let v = vm
        .eval(
            "import { poke_a } from \"user_a\"\nimport { poke_b } from \"user_b\"\nimport { current } from \"counter\"\npoke_a()\npoke_b()\npoke_a()\ncurrent()",
        )
        .unwrap();
    assert_eq!(v, int(3));
}

#[test]
fn module_cannot_see_main_globals() {
    let mut vm = vm_with(&[("leaky", "export fn peek() = main_secret")]);
    vm.eval("let main_secret = 5").unwrap();
    let err = vm
        .eval("import { peek } from \"leaky\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown variable `main_secret`"), "{}", err);
    assert!(err.contains("import"), "should hint at imports: {}", err);
}

#[test]
fn modules_cannot_see_each_others_internals() {
    let mut vm = vm_with(&[
        ("a", "export fn fa() = 1\nlet internal_a = 9"),
        ("b", "export fn fb() = internal_a"),
    ]);
    vm.eval("import { fa } from \"a\"").unwrap();
    let err = vm.eval("import { fb } from \"b\"").unwrap_err().to_string();
    assert!(err.contains("unknown variable `internal_a`"), "{}", err);
}

#[test]
fn modules_see_prelude_and_natives() {
    let mut vm = Funct::new();
    vm.register1("host_inc", |x: i64| x + 1); // registered AFTER Funct::new
    vm.set_module_root(module_root(&[(
        "uses_both",
        "export fn go(xs) = xs |> map(x => host_inc(x)) |> sum",
    )]));
    assert_eq!(
        vm.eval("import { go } from \"uses_both\"\ngo([1, 2, 3])")
            .unwrap(),
        int(9)
    );
}

#[test]
fn module_private_functions_work_internally() {
    let mut vm = vm_with(&[(
        "m",
        "fn helper(x) = x * 10\nexport fn use_helper(x) = helper(x) + 1",
    )]);
    assert_eq!(
        vm.eval("import { use_helper } from \"m\"\nuse_helper(4)")
            .unwrap(),
        int(41)
    );
}

#[test]
fn module_ufcs_resolves_module_locally() {
    // `x.helper()` inside a module must find the module's own helper
    let mut vm = vm_with(&[("m", "fn helper(x) = x * 10\nexport fn go(x) = x.helper()")]);
    assert_eq!(vm.eval("import { go } from \"m\"\ngo(4)").unwrap(), int(40));
}

#[test]
fn host_module_named_import() {
    let mut vm = Funct::new();
    vm.register3("lerp_impl", |a: f64, b: f64, t: f64| a + (b - a) * t);
    let lerp = vm.native_fn("lerp_impl").unwrap();
    vm.register_module("math", vec![("lerp", lerp)]);
    assert_eq!(
        vm.eval("import { lerp } from \"math\"\nlerp(0.0, 10.0, 0.25)")
            .unwrap(),
        Value::Float(2.5)
    );
}

#[test]
fn host_module_qualified_import() {
    let mut vm = Funct::new();
    vm.register1("abs_impl", |x: f64| x.abs());
    let abs = vm.native_fn("abs_impl").unwrap();
    vm.register_module("mathx", vec![("abs", abs)]);
    assert_eq!(
        vm.eval("import \"mathx\" as mx\nmx.abs(0.0 - 4.0)")
            .unwrap(),
        Value::Float(4.0)
    );
}

#[test]
fn host_module_missing_export_fails_loudly() {
    let mut vm = Funct::new();
    vm.register_module("m", vec![]);
    let err = vm
        .eval("import { nope } from \"m\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no export `nope`"), "{}", err);
}

#[test]
fn import_inside_block_is_rejected() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let err = vm
        .eval("fn f() {\n import { double } from \"math\"\n 1\n}")
        .unwrap_err()
        .to_string();
    assert!(err.contains("top level"), "{}", err);
}

#[test]
fn import_path_traversal_is_rejected() {
    let mut vm = vm_with(&[]);
    let err = vm
        .eval("import { x } from \"../evil\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid module path"), "{}", err);
}

#[test]
fn hot_reload_module_functions() {
    let root = module_root(&[("m", "export fn version() = 1")]);
    let mut vm = Funct::new();
    vm.set_module_root(&root);
    vm.eval("import { version } from \"m\"").unwrap();
    assert_eq!(vm.eval("version()").unwrap(), int(1));
    // rewrite the file and reload: the already-imported binding hot-swaps
    std::fs::write(root.join("m.ft"), "export fn version() = 2").unwrap();
    vm.reload_module("m").unwrap();
    assert_eq!(vm.eval("version()").unwrap(), int(2));
}

#[test]
fn qualified_module_sees_reloaded_functions_too() {
    let root = module_root(&[("m", "export fn version() = 1")]);
    let mut vm = Funct::new();
    vm.set_module_root(&root);
    vm.eval("import \"m\" as m").unwrap();
    assert_eq!(vm.eval("m.version()").unwrap(), int(1));
    std::fs::write(root.join("m.ft"), "export fn version() = 2").unwrap();
    vm.reload_module("m").unwrap();
    // the record holds a Closure with a stable fn id -> new code
    assert_eq!(vm.eval("m.version()").unwrap(), int(2));
}

#[test]
fn module_state_survives_in_snapshot() {
    // a snapshot is self-contained: restore + use imports WITHOUT the files
    let counter_mod = r#"
let count = atom(100)
export fn bump() = swap!(count, n => n + 1)
"#;
    let mut vm = vm_with(&[("counter", counter_mod)]);
    vm.eval("import { bump } from \"counter\"\nbump()").unwrap();
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let json = vm.save_state(&st).unwrap();

    let mut vm2 = Funct::new(); // note: NO module root, no files
    vm2.restore_state(&json).unwrap();
    assert_eq!(vm2.eval("bump()").unwrap(), int(102));
}

#[test]
fn export_expression_is_rejected() {
    let mut vm = Funct::new();
    let err = vm.eval("export 1 + 1").unwrap_err().to_string();
    assert!(err.contains("exported"), "{}", err);
}

#[test]
fn module_top_level_faults_surface() {
    let mut vm = vm_with(&[("bad", "export fn f() = 1\nlet boom = 1 / 0")]);
    let err = vm
        .eval("import { f } from \"bad\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("division by zero"), "{}", err);
}

#[test]
fn load_module_api_lists_exports() {
    let mut vm = vm_with(&[("math", MATH_MOD)]);
    let mut exports = vm.load_module("math").unwrap();
    exports.sort();
    assert_eq!(
        exports,
        vec![
            "answer".to_string(),
            "double".to_string(),
            "triple".to_string()
        ]
    );
}

#[test]
fn exported_let_with_destructuring() {
    let mut vm = vm_with(&[("cfg", "export let (host, port) = (\"localhost\", 8080)")]);
    assert_eq!(
        vm.eval("import { host, port } from \"cfg\"\n\"${host}:${port}\"")
            .unwrap(),
        Value::str("localhost:8080")
    );
}

// A host-interface module declares `extern` natives the embedding host
// provides. Those externs are part of the module's export surface, so they
// resolve through every import form, not just bare `import "host"`.
const HOST_IFACE: &str = r#"
extern let canvas_w
extern fn mask_paint(name, x, y, radius, value)
"#;

#[test]
fn externs_are_importable_by_name() {
    let mut vm = vm_with(&[("host", HOST_IFACE)]);
    vm.register5(
        "mask_paint",
        |_n: String, _x: f64, _y: f64, r: f64, _v: f64| r as i64,
    );
    vm.set_global("canvas_w", int(800));
    // named import of an extern fn + an extern let
    assert_eq!(
        vm.eval("import { mask_paint, canvas_w } from \"host\"\nmask_paint(\"r\", 0.0, 0.0, 48.0, 0.0) + canvas_w")
            .unwrap(),
        int(848)
    );
}

#[test]
fn externs_are_importable_with_alias_and_qualified() {
    let mut vm = vm_with(&[("host", HOST_IFACE)]);
    vm.register5(
        "mask_paint",
        |_n: String, _x: f64, _y: f64, r: f64, _v: f64| r as i64,
    );
    vm.set_global("canvas_w", int(800));
    assert_eq!(
        vm.eval("import { mask_paint as paint } from \"host\"\npaint(\"r\", 0.0, 0.0, 7.0, 0.0)")
            .unwrap(),
        int(7)
    );
    let mut vm = vm_with(&[("host", HOST_IFACE)]);
    vm.register5(
        "mask_paint",
        |_n: String, _x: f64, _y: f64, r: f64, _v: f64| r as i64,
    );
    vm.set_global("canvas_w", int(800));
    assert_eq!(
        vm.eval("import \"host\" as h\nh.canvas_w").unwrap(),
        int(800)
    );
}

#[test]
fn unprovided_imported_extern_faults_loudly_when_called() {
    let mut vm = vm_with(&[("host", HOST_IFACE)]);
    // host registers nothing: the named import still loads, but calling faults
    let err = vm
        .eval("import { mask_paint } from \"host\"\nmask_paint(\"r\", 0.0, 0.0, 1.0, 0.0)")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("declared `extern` but this host does not provide it"),
        "{}",
        err
    );
}

// ---------- host modules require explicit import (no bare/leaked access) ----------

fn lerp_fn(vm: &mut Funct) -> Value {
    vm.register3("lerp_impl", |a: f64, b: f64, t: f64| a + (b - a) * t);
    vm.native_fn("lerp_impl").unwrap()
}

#[test]
fn host_module_named_import_with_alias() {
    let mut vm = Funct::new();
    let lerp = lerp_fn(&mut vm);
    vm.register_module("math", vec![("lerp", lerp)]);
    assert_eq!(
        vm.eval("import { lerp as mix } from \"math\"\nmix(0.0, 10.0, 0.5)")
            .unwrap(),
        Value::Float(5.0)
    );
}

#[test]
fn host_module_not_reachable_by_bare_name() {
    let mut vm = Funct::new();
    let lerp = lerp_fn(&mut vm);
    vm.register_module("math", vec![("lerp", lerp)]);
    // No import: neither the bare module nor a field access resolves.
    assert!(vm.eval("math").is_err());
    assert!(vm.eval("math.lerp(0.0, 1.0, 0.5)").is_err());
}

#[test]
fn host_module_not_visible_inside_a_file_module_without_import() {
    // A .ft module that references a host module without importing it must
    // fail to compile, exactly like referencing any other un-imported global.
    let mut vm = vm_with(&[("usesmath", "export fn go() = math.lerp(0.0, 10.0, 0.5)")]);
    let lerp = lerp_fn(&mut vm);
    vm.register_module("math", vec![("lerp", lerp)]);
    let err = vm
        .eval("import { go } from \"usesmath\"\ngo()")
        .unwrap_err()
        .to_string();
    assert!(err.contains("math"), "{}", err);
}

#[test]
fn re_registering_a_host_module_replaces_its_contents() {
    let mut vm = Funct::new();
    let lerp = lerp_fn(&mut vm);
    vm.register_module("math", vec![("lerp", lerp)]);
    // re-register with a different surface
    vm.register1("neg_impl", |x: f64| -x);
    let neg = vm.native_fn("neg_impl").unwrap();
    vm.register_module("math", vec![("neg", neg)]);
    assert_eq!(
        vm.eval("import { neg } from \"math\"\nneg(3.0)").unwrap(),
        Value::Float(-3.0)
    );
    // the old export is gone
    assert!(vm.eval("import { lerp } from \"math\"").is_err());
}

#[test]
fn importing_unregistered_module_name_fails_loudly() {
    let mut vm = Funct::new();
    let err = vm
        .eval("import { x } from \"nope_not_a_module\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("nope_not_a_module"), "{}", err);
}
