//! Verify memoization codegen against Node: the emitted code must (1) compute
//! the same result as the original for every input, and (2) actually memoize —
//! reuse cached object references when dependencies are unchanged.

use jsir_ssa::{codegen, lower, mutability, scopes, ssa};

fn node_available() -> bool {
    std::process::Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

const TAG_JS: &str = r#"
function __tag(v){
  if (v === undefined) return "u:";
  if (v === null) return "l:";
  if (typeof v === "boolean") return "b:"+v;
  if (typeof v === "number") return "n:"+(Number.isNaN(v)?"NaN":String(v));
  if (typeof v === "string") return "s:"+v;
  if (Array.isArray(v)) return "a:["+v.map(__tag).join(",")+"]";
  if (typeof v === "object"){var ks=Object.keys(v).sort();return "o:{"+ks.map(function(k){return k+"="+__tag(v[k]);}).join(",")+"}";}
  return "?:"+typeof v;
}
"#;

fn run_node(program: &str) -> Option<String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!("jsir_codegen_{}_{}.js", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
    std::fs::File::create(&path).ok()?.write_all(program.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        eprintln!("node error: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn memoize(src: &str) -> String {
    let mut cfg = lower(src).expect("lower");
    ssa::construct(&mut cfg);
    let r = mutability::analyze(&cfg);
    let infos = scopes::analyze(&cfg, &r);
    codegen::emit_memoized(&cfg, &infos, "C").expect("emit")
}

#[test]
fn memoized_matches_original() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let cases: &[(&str, Vec<Vec<&str>>)] = &[
        ("function C(a, b){ let style={color:a}; let el={size:b, props:style}; return el; }",
         vec![vec!["1", "2"], vec!["3", "3"], vec!["7", "9"]]),
        ("function C(x, y){ let p={a:x+y, b:x-y}; let q={p:p, sum:x+y}; return q; }",
         vec![vec!["2", "1"], vec!["5", "5"]]),
        ("function C(n){ let xs=[n, n+1, n+2]; let o={list:xs, head:n}; return o; }",
         vec![vec!["0"], vec!["4"], vec!["9"]]),
    ];
    let mut checked = 0;
    for (src, inputs) in cases {
        let memo = memoize(src);
        for args in inputs {
            let arglist = args.join(", ");
            let orig = run_node(&format!("{TAG_JS}\n{src}\nconsole.log(__tag(C({arglist})));"));
            let mem = run_node(&format!("{}{TAG_JS}\n{memo}\nconsole.log(__tag(C({arglist})));", codegen::RUNTIME));
            assert_eq!(orig, mem, "[{src}] C({arglist})\n--- memoized ---\n{memo}");
            checked += 1;
        }
    }
    eprintln!("memoized_matches_original: {checked} (component,input) pairs match node");
}

#[test]
fn memoization_actually_caches() {
    if !node_available() {
        return;
    }
    let src = "function C(a, b){ let style={color:a}; let el={size:b, props:style}; return el; }";
    let memo = memoize(src);
    // Persistent cache across calls (simulating React's per-instance useMemoCache).
    let persistent = "const _e = Symbol('e'); let __c=null; function _c(n){ if(!__c) __c=new Array(n).fill(_e); return __c; }\n";
    let driver = format!(
        "{persistent}{memo}\n\
         const r1 = C(1, 2);\n\
         const r2 = C(1, 2);\n\
         const r3 = C(1, 5);\n\
         const r4 = C(9, 5);\n\
         console.log([r1===r2, r1.props===r3.props, r1!==r3, r3.props!==r4.props].join(','));\n"
    );
    let got = run_node(&driver).expect("run");
    // r1===r2: same deps -> el reused.
    // r1.props===r3.props: `a` unchanged -> style reused even though b changed.
    // r1!==r3: b changed -> el recomputed.
    // r3.props!==r4.props: `a` changed -> style recomputed.
    assert_eq!(got, "true,true,true,true", "memoization stability\n{memo}");
}
