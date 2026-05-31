//! Executable oracle for the CFG + SSA layer.
//!
//! For each program and each input tuple we check that **three** results agree:
//!   1. Node running the original JavaScript,
//!   2. our interpreter on the pre-SSA CFG,
//!   3. our interpreter on the SSA CFG,
//! and that the SSA form passes the dominance verifier. Any lowering or SSA bug
//! surfaces as a value mismatch against Node.

use jsir_ssa::interp::{self, Val};
use jsir_ssa::{lower, ssa, verify};

fn node_available() -> bool {
    std::process::Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// JS `tag()` mirroring `interp::tag`, plus a driver that prints `tag(f(args))`.
const TAG_JS: &str = r#"
function __tag(v){
  if (v === undefined) return "u:";
  if (v === null) return "l:";
  if (typeof v === "boolean") return "b:"+v;
  if (typeof v === "number") return "n:"+(Number.isNaN(v)?"NaN":v===Infinity?"Infinity":v===-Infinity?"-Infinity":String(v));
  if (typeof v === "string") return "s:"+v;
  if (Array.isArray(v)) return "a:["+v.map(__tag).join(",")+"]";
  if (typeof v === "object"){var ks=Object.keys(v).sort();return "o:{"+ks.map(function(k){return k+"="+__tag(v[k]);}).join(",")+"}";}
  return "?:"+typeof v;
}
"#;

fn js_literal(v: &Val) -> String {
    match v {
        Val::Undef => "undefined".into(),
        Val::Null => "null".into(),
        Val::Bool(b) => b.to_string(),
        Val::Num(n) => {
            if n.is_nan() {
                "NaN".into()
            } else if n.is_infinite() {
                if *n > 0.0 { "Infinity".into() } else { "-Infinity".into() }
            } else {
                interp::js_num_to_string(*n)
            }
        }
        Val::Str(s) => format!("{s:?}"),
        _ => unreachable!("args are primitives"),
    }
}

fn run_node(src: &str, args: &[Val]) -> Option<String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let arglist = args.iter().map(js_literal).collect::<Vec<_>>().join(", ");
    let program = format!("{TAG_JS}\n{src}\nconsole.log(__tag(f({arglist})));\n");
    let dir = std::env::temp_dir();
    let path = dir.join(format!("jsir_ssa_{}_{}.js", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
    std::fs::File::create(&path).ok()?.write_all(program.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The full check for one program + one input tuple.
fn check_one(src: &str, args: &[Val]) -> Result<(), String> {
    let mut cfg = lower(src).map_err(|e| format!("lower: {e}"))?;
    let pre = interp::run(&cfg, args).map_err(|e| format!("pre-ssa interp: {e}"))?;
    let pre_tag = interp::tag(&pre);

    ssa::construct(&mut cfg);
    let errs = verify::verify(&cfg);
    if !errs.is_empty() {
        return Err(format!("verify failed: {}", errs.join("; ")));
    }
    let post = interp::run(&cfg, args).map_err(|e| format!("ssa interp: {e}"))?;
    let post_tag = interp::tag(&post);

    if pre_tag != post_tag {
        return Err(format!("pre-ssa {pre_tag} != ssa {post_tag}"));
    }

    if let Some(node_tag) = run_node(src, args) {
        if node_tag != post_tag {
            return Err(format!("node {node_tag} != ours {post_tag}"));
        }
    }
    Ok(())
}

const NUM_INPUTS: &[f64] = &[-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];

/// Programs with a single numeric param `n`.
const UNARY: &[&str] = &[
    "function f(n){ let s=0; let i=0; while(i<n){ s=s+i; i=i+1; } return s; }",
    "function f(n){ let r=1; let i=1; while(i<=n){ r=r*i; i=i+1; } return r; }", // factorial
    "function f(n){ let s=0; let i=0; while(i<n){ if(i%2===0){ s=s+i; } i=i+1; } return s; }",
    "function f(x){ if(x<0){ return -1; } if(x>0){ return 1; } return 0; }",
    "function f(x){ x=x+1; x=x*2; return x; }",
    "function f(x){ let y; if(x>0){ y=1; } else { y=2; } return y; }",
    "function f(n){ let s=0; let i=0; while(i<n){ s+=i; i+=1; } return s; }",
    "function f(n){ let s=''; let i=0; while(i<n){ s=s+i; i=i+1; } return s; }",
    "function f(n){ let t=0; let i=0; while(i<n){ let j=0; while(j<i){ t=t+1; j=j+1; } i=i+1; } return t; }",
    "function f(x){ let y=x; while(y>1){ if(y%2===0){ y=y/2; } else { y=y*3+1; } } return y; }", // collatz -> 1
    "function f(n){ let a=0; let b=1; let i=0; while(i<n){ let t=a+b; a=b; b=t; i=i+1; } return a; }", // fib
];

/// Programs with two numeric params `a, b`.
const BINARY: &[&str] = &[
    "function f(a,b){ return a>b ? a : b; }",
    "function f(a,b){ let m=a; if(b>m){ m=b; } return m; }",
    "function f(a,b){ let s=0; let i=a; while(i<b){ s=s+i; i=i+1; } return s; }",
    "function f(a,b){ return (a+b)*(a-b); }",
    "function f(a,b){ if(b<=0){ return 0; } let q=0; let r=a; while(r>=b){ r=r-b; q=q+1; } return q; }", // integer division (guarded)
];

#[test]
fn oracle_unary() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let mut checked = 0;
    for src in UNARY {
        for &n in NUM_INPUTS {
            if let Err(e) = check_one(src, &[Val::Num(n)]) {
                panic!("[{src}] f({n}): {e}");
            }
            checked += 1;
        }
    }
    eprintln!("oracle_unary: {checked} (program,input) pairs agreed with node");
}

#[test]
fn oracle_binary() {
    if !node_available() {
        return;
    }
    let mut checked = 0;
    for src in BINARY {
        for &a in NUM_INPUTS {
            for &b in NUM_INPUTS {
                if let Err(e) = check_one(src, &[Val::Num(a), Val::Num(b)]) {
                    panic!("[{src}] f({a},{b}): {e}");
                }
                checked += 1;
            }
        }
    }
    eprintln!("oracle_binary: {checked} (program,input) pairs agreed with node");
}

/// Programs exercising objects/arrays + member reads/writes (mutation), with a
/// single numeric param `n`.
const OBJECTS: &[&str] = &[
    "function f(n){ let o={x:0,y:0}; let i=0; while(i<n){ o.x=o.x+i; o.y=o.y+1; i=i+1; } return o.x+o.y; }",
    "function f(n){ let o={c:0}; let i=0; while(i<n){ if(i%2===0){ o.c=o.c+1; } i=i+1; } return o.c; }",
    "function f(n){ let arr=[1,2,3]; arr[0]=n; return arr[0]+arr[1]+arr[2]; }",
    "function f(n){ let a=[]; let i=0; while(i<n){ a[i]=i*i; i=i+1; } let s=0; let j=0; while(j<n){ s=s+a[j]; j=j+1; } return s; }",
    "function f(n){ let o={v:n}; o.v=o.v*2; o.w=o.v+1; return o.v+o.w; }",
];

#[test]
fn oracle_objects() {
    if !node_available() {
        return;
    }
    let mut checked = 0;
    for src in OBJECTS {
        for &n in &[0.0, 1.0, 2.0, 3.0, 5.0, 8.0] {
            if let Err(e) = check_one(src, &[Val::Num(n)]) {
                panic!("[{src}] f({n}): {e}");
            }
            checked += 1;
        }
    }
    eprintln!("oracle_objects: {checked} (program,input) pairs agreed with node");
}

/// The SSA form must always pass the verifier, independent of node.
#[test]
fn ssa_well_formed() {
    for src in UNARY.iter().chain(BINARY).chain(OBJECTS) {
        let mut cfg = lower(src).expect("lower");
        ssa::construct(&mut cfg);
        let errs = verify::verify(&cfg);
        assert!(errs.is_empty(), "[{src}] verify: {}", errs.join("; "));
    }
}
