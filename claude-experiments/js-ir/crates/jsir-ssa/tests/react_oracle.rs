//! **React Compiler as an oracle.** We compile a component with our pipeline and
//! with the real `react-compiler-e2e` CLI (the Rust port, cloned alongside), then
//! compare the *memoization structure* they each decide on — the `_c(N)` cache
//! size and the number of memo blocks. Those two numbers encode the scope count
//! and total dependency count, so matching them across many components is strong
//! evidence our mutable-range / reactive-scope / merge passes agree with React.
//!
//! The text differs (we emit `createElement`, React keeps JSX; different temp
//! names), so we compare structure, not bytes. Set `REACT_CC` to the CLI binary,
//! or it defaults to the cloned build under /tmp; the test skips if absent.

use jsir_ssa::codegen;

fn react_cli() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("REACT_CC") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let def = std::path::PathBuf::from("/tmp/react-rust/compiler/target/release/react-compiler-e2e");
    def.exists().then_some(def)
}

/// `(cache_size, memo_block_count)` extracted from compiled output.
fn structure(code: &str) -> Option<(usize, usize)> {
    let n = code
        .split("_c(")
        .nth(1)?
        .split(')')
        .next()?
        .trim()
        .parse::<usize>()
        .ok()?;
    // A memo block is `if (` then any number of `(` then a cache check `$[`.
    // (React parenthesizes pairwise, so 3 deps -> `if ((($[...`.)
    let block_count = code
        .match_indices("if (")
        .filter(|(i, _)| code[i + 4..].trim_start_matches(['(', ' ']).starts_with("$["))
        .count();
    Some((n, block_count))
}

fn react_compile(cli: &std::path::Path, src: &str) -> Option<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(cli)
        .args(["--frontend", "swc", "--filename", "t.jsx"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(src.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

/// Components whose memoization structure we match React on exactly.
const COMPONENTS: &[&str] = &[
    "function C(props) { const items = [props.a, props.b, props.c]; return <ul>{items}</ul>; }",
    "function E(props) { const arr = [props.x]; const wrap = {arr: arr, x: props.x}; return <span data={wrap}>{arr}</span>; }",
    "function F(props) { return <div id={props.id} className={props.cls} />; }",
    "function G(props) { const inner = {v: props.v}; const outer = {inner: inner}; return <div d={outer} />; }",
    // Formerly KNOWN_INTERMEDIATE_OBJECT_GAPS: intermediate objects/arrays whose
    // only downstream use is a member read or JSX `children` slot. Now matched
    // exactly (the property-path dependency / intermediate-object memoization
    // landed), so promoted into the matched set.
    "function A(props) { const x = {a: props.a}; return <div>{x.a}</div>; }",
    "function Foo(props) { const style = {color: props.color}; const data = [props.a, props.b]; return <div style={style}>{data}</div>; }",
    "function B(props) { const a = {x: props.p}; const b = {y: props.p, z: a}; return <li>{b}</li>; }",
    "function D(props) { const s = {n: props.n}; const t = {m: props.m}; return <div a={s} b={t} />; }",
];

#[test]
fn matches_react_compiler_structure() {
    let Some(cli) = react_cli() else {
        eprintln!("react-compiler-e2e not found (set REACT_CC); skipping");
        return;
    };

    let mut agree = 0;
    let mut total = 0;
    let mut diffs = Vec::new();
    for src in COMPONENTS {
        let ours = match codegen::compile(src) {
            Ok(c) => c,
            Err(e) => {
                diffs.push(format!("OURS failed to compile: {e}\n  {src}"));
                continue;
            }
        };
        let Some(react) = react_compile(&cli, src) else {
            diffs.push(format!("react CLI failed:\n  {src}"));
            continue;
        };
        total += 1;
        match (structure(&ours), structure(&react)) {
            (Some(o), Some(r)) if o == r => agree += 1,
            (o, r) => diffs.push(format!(
                "structure mismatch ours={o:?} react={r:?}\n  {src}\n--- ours ---\n{ours}\n--- react ---\n{react}"
            )),
        }
    }

    eprintln!("react-oracle: {agree}/{total} components matched React's (cache size, scope count)");
    assert!(diffs.is_empty(), "{}", diffs.join("\n\n"));
    assert!(total > 0, "no components were compared");
}
