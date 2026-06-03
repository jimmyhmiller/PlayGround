//! One-time dumper of React's (oracle) structure per fixture: `<name>\t<struct>`
//! where struct is `cache,blocks` / `none` (compiles, no memo) / `fail` (errors).
//! React's verdict is fixed, so capture it ONCE and reuse as the stable
//! reference for fast local agree/mismatch computation (join against cc_batch).
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
        let verdict = (|| {
            let mut child = Command::new(&cli)
                .args(["--frontend", "swc", "--filename", "t.jsx"])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
                .spawn().ok()?;
            child.stdin.take()?.write_all(src.as_bytes()).ok()?;
            let out = child.wait_with_output().ok()?;
            if !out.status.success() { return Some("fail".to_string()); }
            let code = String::from_utf8_lossy(&out.stdout);
            Some(structure(&code).map(|(c, b)| format!("{c},{b}")).unwrap_or_else(|| "none".into()))
        })().unwrap_or_else(|| "fail".to_string());
        println!("{name}\t{verdict}");
    }
}
