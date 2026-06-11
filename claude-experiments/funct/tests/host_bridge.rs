//! Host-bridge surface for embedding in an editor: set_global injection,
//! serde_json::Value conversions across the boundary, and the widget-style
//! handler-dispatch pattern end to end.

use funct::{Funct, Value};
use serde_json::json;

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn set_global_injects_host_values() {
    let mut vm = Funct::new();
    vm.set_global("canvas_w", Value::Float(800.0));
    vm.set_global("canvas_h", Value::Float(600.0));
    assert_eq!(vm.eval("canvas_w / canvas_h").unwrap(), Value::Float(800.0 / 600.0));
    // host can update it between calls
    vm.set_global("canvas_w", Value::Float(1024.0));
    assert_eq!(vm.eval("canvas_w").unwrap(), Value::Float(1024.0));
}

#[test]
fn set_globals_are_visible_in_modules() {
    let dir = std::env::temp_dir().join(format!("funct_hostbridge_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.ft"), "export fn double_w() = canvas_w * 2").unwrap();
    let mut vm = Funct::new();
    vm.set_module_root(&dir);
    vm.set_global("canvas_w", Value::Float(100.0));
    assert_eq!(
        vm.eval("import { double_w } from \"m\"\ndouble_w()").unwrap(),
        Value::Float(200.0)
    );
}

#[test]
fn json_value_to_script_and_back() {
    let mut vm = Funct::new();
    // host -> script: a bus payload arrives as JSON
    let payload = json!({ "tool": "Edit", "count": 3, "tags": ["a", "b"], "extra": null });
    vm.set_global("payload", Value::from_json(&payload));
    assert_eq!(vm.eval("payload.tool").unwrap(), Value::str("Edit"));
    assert_eq!(vm.eval("payload.count + 1").unwrap(), int(4));
    assert_eq!(vm.eval("len(payload.tags)").unwrap(), int(2));
    assert_eq!(vm.eval("payload.extra").unwrap(), Value::Unit);

    // script -> host: a render-frame-ish record comes back as JSON
    let v = vm
        .eval("{ kind: \"column\", children: [{ kind: \"text\", value: \"hi\" }], pad: 4.5 }")
        .unwrap();
    let j = v.to_json().unwrap();
    assert_eq!(j["kind"], "column");
    assert_eq!(j["children"][0]["value"], "hi");
    assert_eq!(j["pad"], 4.5);
}

#[test]
fn json_in_registered_fn_signatures() {
    let mut vm = Funct::new();
    // natives can take/return serde_json::Value directly
    vm.register1("echo_json", |v: serde_json::Value| v);
    assert_eq!(
        vm.eval("echo_json({ a: [1, 2] })").unwrap(),
        vm.eval("{ a: [1, 2] }").unwrap()
    );
}

#[test]
fn to_json_fails_loudly_on_unrepresentable() {
    let mut vm = Funct::new();
    let v = vm.eval("atom(1)").unwrap();
    let err = v.to_json().unwrap_err();
    assert!(err.msg.contains("cannot convert Atom"), "{}", err.msg);
    let f = vm.eval("x => x").unwrap();
    assert!(f.to_json().is_err());
}

#[test]
fn tuples_become_json_arrays() {
    let mut vm = Funct::new();
    let v = vm.eval("(1, \"a\")").unwrap();
    assert_eq!(v.to_json().unwrap(), json!([1, "a"]));
}

/// The widget pattern end to end: state in an atom, named handlers invoked
/// by the host, render() returning a frame the host reads as JSON.
#[test]
fn widget_style_handler_dispatch() {
    let script = r#"
let state = atom({ count: 0, label: "ready" })

export fn on_click(x, y) {
    swap_in!(state, ["count"], n => n + 1)
    reset_in!(state, ["label"], "clicked at ${x},${y}")
}

export fn render(w, h) {
    { kind: "text", value: "${(@state).label} (${(@state).count})", w: w, h: h }
}
"#;
    let mut vm = Funct::new();
    vm.eval(script).unwrap();

    // host dispatches events by name — missing handlers are simply absent
    assert!(vm.global("on_hover").is_none());
    assert!(vm.global("on_click").is_some());

    vm.call("on_click", vec![Value::Float(10.0), Value::Float(20.0)]).unwrap();
    vm.call("on_click", vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();

    let frame = vm.call("render", vec![Value::Float(640.0), Value::Float(480.0)]).unwrap();
    let j = frame.to_json().unwrap();
    assert_eq!(j["kind"], "text");
    assert_eq!(j["value"], "clicked at 1.0,2.0 (2)");
    assert_eq!(j["w"], 640.0);

    // and the whole widget (code + atom state) snapshots to disk
    let st = funct::VmState { frames: vec![], stack: vec![], status: funct::Status::Done(Value::Unit) };
    let saved = vm.save_state(&st).unwrap();
    let mut vm2 = Funct::new();
    vm2.restore_state(&saved).unwrap();
    vm2.call("on_click", vec![Value::Float(0.0), Value::Float(0.0)]).unwrap();
    let frame2 = vm2.call("render", vec![Value::Float(100.0), Value::Float(100.0)]).unwrap();
    assert_eq!(frame2.to_json().unwrap()["value"], "clicked at 0.0,0.0 (3)");
}
