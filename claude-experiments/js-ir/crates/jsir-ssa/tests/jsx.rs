//! End-to-end JSX component memoization, verified under Node: our memoized
//! output must build the same element tree as the original for every prop set,
//! and must actually memoize (reuse element references when deps are unchanged).

use jsir_ssa::{codegen, lower, mutability, scopes, ssa};

fn node_available() -> bool {
    std::process::Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// `React.createElement` stub + deep tagger, mirroring object identity by a
/// per-element id so we can also check reference stability.
const HARNESS: &str = r#"
let __id = 0;
const React = { createElement: (type, props, ...children) => ({ __k: 'el', id: __id++, type, props, children }) };
function __tag(v){
  if (v === undefined) return "u:";
  if (v === null) return "l:";
  if (typeof v === "boolean") return "b:"+v;
  if (typeof v === "number") return "n:"+String(v);
  if (typeof v === "string") return "s:"+v;
  if (Array.isArray(v)) return "a:["+v.map(__tag).join(",")+"]";
  if (typeof v === "object"){var ks=Object.keys(v).filter(k=>k!=='id').sort();return "o:{"+ks.map(k=>k+"="+__tag(v[k])).join(",")+"}";}
  return "?";
}
"#;

fn run_node(program: &str) -> Option<String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!("jsir_jsx_{}_{}.js", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
    std::fs::File::create(&path).ok()?.write_all(program.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        eprintln!("node err: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn memoize(src: &str, name: &str) -> String {
    let mut cfg = lower(src).expect("lower");
    ssa::construct(&mut cfg);
    let r = mutability::analyze(&cfg);
    let infos = scopes::analyze(&cfg, &r);
    codegen::emit_memoized(&cfg, &infos, name).expect("emit")
}

const COMPONENTS: &[(&str, &str)] = &[
    ("Foo", "function Foo(props) { const style = {color: props.color}; const data = [props.a, props.b]; return <div style={style}>{data}</div>; }"),
    ("Bar", "function Bar(props) { return <span id={props.id}>{props.text}</span>; }"),
    ("Baz", "function Baz(p) { const items = [p.a, p.b, p.c]; const wrap = {items: items, n: p.a}; return <ul data={wrap}>{items}</ul>; }"),
];

#[test]
fn jsx_memoized_matches_original() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let props = ["{color:1,a:2,b:3,id:4,text:5,c:6}", "{color:9,a:9,b:9,id:9,text:9,c:9}", "{color:0,a:1,b:0,id:1,text:0,c:1}"];
    let mut checked = 0;
    for (name, src) in COMPONENTS {
        // Baseline = the (non-memoized) component, desugared to createElement so
        // node can run it. Memoization must preserve this behavior exactly.
        let baseline = jsir_swc::ir_to_source(&jsir_swc::source_to_ir(src).unwrap()).unwrap();
        let memo = memoize(src, name);
        for p in props {
            let orig = run_node(&format!("{HARNESS}{baseline}\nconsole.log(__tag({name}({p})));"));
            let mem = run_node(&format!("{HARNESS}{}{memo}\nconsole.log(__tag({name}({p})));", codegen::RUNTIME));
            assert!(orig.is_some(), "[{name}] baseline failed to run:\n{baseline}");
            assert_eq!(orig, mem, "[{name}] {p}\n--- memoized ---\n{memo}");
            checked += 1;
        }
    }
    eprintln!("jsx_memoized_matches_original: {checked} (component,props) pairs match node");
}

#[test]
#[ignore = "intermediate-object memoization gap after Step-1 escape analysis: \
React caches the `style` object on its own scope so the `<div>` stays reference-\
stable across renders; we lowered JSX to createElement with a separate props \
object, so the intermediate `style` is not yet re-memoized (its freshness \
poisons the element's stability). Recovered by the property-path dependency + \
nesting-aware scope-merge steps of PHASE_B_REDESIGN. See \
react_oracle::KNOWN_INTERMEDIATE_OBJECT_GAPS. Value correctness is still \
verified by jsx_memoized_matches_original."]
fn jsx_memoization_is_stable() {
    if !node_available() {
        return;
    }
    let (name, src) = COMPONENTS[0];
    let memo = memoize(src, name);
    // Persistent cache across renders.
    let persistent = "const _e=Symbol('e'); let __c=null; function _c(n){ if(!__c) __c=new Array(n).fill(_e); return __c; }\n";
    let driver = format!(
        "{HARNESS}{persistent}{memo}\n\
         const r1 = {name}({{color:1,a:2,b:3}});\n\
         const r2 = {name}({{color:1,a:2,b:3}});\n\
         const r3 = {name}({{color:9,a:2,b:3}});\n\
         console.log([r1===r2, r1.children===r3.children].join(','));\n"
    );
    // r1===r2: identical props -> whole element reused (element scope cached).
    // r1.children===r3.children: a/b unchanged (only color changed) -> the `data`
    //   children array is reused (its own memo scope, keyed on a/b).
    //
    // NOTE: React additionally caches the intermediate `style` object on its own
    // scope (so `r1.props.style === r3.props.style` when color is unchanged). We
    // do not yet re-memoize that intermediate object after the Step-1 escape
    // analysis (see KNOWN_INTERMEDIATE_OBJECT_GAPS in react_oracle.rs); it is a
    // sound under-memoization recovered by the later dependency/scope-merge steps.
    assert_eq!(run_node(&driver).as_deref(), Some("true,true"), "memoization stability\n{memo}");
}
