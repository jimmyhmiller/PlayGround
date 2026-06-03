//! Analysis probe (NOT the gate): dump every fixture where both we and React
//! memoize but the structure differs, with the (cache_size, block_count) delta.
//! Lets us bucket mismatches by cause. Reuses the same react-cc.js oracle the
//! corpus harness uses (env REACT_CC / REACT_FIXTURES).
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn structure(code: &str) -> Option<(usize, usize)> {
    let n = code.split("_c(").nth(1)?.split(')').next()?.trim().parse::<usize>().ok()?;
    let block_count = code
        .match_indices("if (")
        .filter(|(i, _)| code[i + 4..].trim_start_matches(['(', ' ']).starts_with("$["))
        .count();
    Some((n, block_count))
}

fn run_react(cli: &str, src: &str) -> Option<(usize, usize)> {
    let mut child = Command::new(cli)
        .args(["--frontend", "swc", "--filename", "t.jsx"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
        .spawn().ok()?;
    child.stdin.take()?.write_all(src.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() { return None; }
    structure(&String::from_utf8_lossy(&out.stdout))
}

fn main() {
    let fixtures: PathBuf = std::env::var("REACT_FIXTURES").expect("REACT_FIXTURES").into();
    let cli = std::env::var("REACT_CC").expect("REACT_CC");
    let mut paths: Vec<_> = std::fs::read_dir(&fixtures).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "js").unwrap_or(false))
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = match std::fs::read_to_string(&path) { Ok(s) => s, Err(_) => continue };
        let ours = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            jsir_ssa::codegen::compile(&src).ok().and_then(|c| structure(&c))
        })).ok().flatten();
        let ours = match ours { Some(o) => o, None => continue }; // only where WE memoize
        let react = match run_react(&cli, &src) { Some(r) => r, None => continue }; // only where React memoizes
        if ours != react {
            let (rc, rb) = react; let (oc, ob) = ours;
            let kind = if oc > rc { "OVER " } else if oc < rc { "UNDER" } else { "BLOCK" };
            println!("{kind} react=({rc},{rb}) ours=({oc},{ob}) d_cache={} {name}", oc as i64 - rc as i64);
        }
    }
}
