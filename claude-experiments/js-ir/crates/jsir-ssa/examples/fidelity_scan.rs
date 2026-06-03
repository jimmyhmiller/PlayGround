//! Enumerate every base-IR op-kind that `source_to_ir` emits across the React
//! fixture corpus, and report how many distinct fixtures each appears in. This
//! is the raw input to the static fidelity audit (`src/fidelity.rs`): the audit
//! registry must classify exactly the op-kinds this scan finds reachable.
//!
//! Run: `cargo run -q -p jsir-ssa --example fidelity_scan`
//! Optional: `REACT_FIXTURES=/path/to/fixtures/compiler`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use jsir_ir::Op as IrOp;

fn fixtures_dir() -> PathBuf {
    if let Ok(d) = std::env::var("REACT_FIXTURES") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(format!(
        "{home}/Documents/Code/open-source/react-rust-pr36173/compiler/packages/babel-plugin-react-compiler/src/__tests__/fixtures/compiler"
    ))
}

/// Recursively collect every op `name` reachable from `op`.
fn collect(op: &IrOp, out: &mut Vec<String>) {
    out.push(op.name.clone());
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                collect(o, out);
            }
        }
    }
}

fn main() {
    let dir = fixtures_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read fixtures {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "js").unwrap_or(false))
        .collect();
    files.sort();

    // op-kind -> (#fixtures it appears in, total occurrences)
    let mut seen: BTreeMap<String, (u32, u64)> = BTreeMap::new();
    let mut parsed = 0u32;
    let mut parse_fail = 0u32;

    for p in &files {
        let src = match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let ir = match jsir_swc::source_to_ir(&src) {
            Ok(ir) => ir,
            Err(_) => {
                parse_fail += 1;
                continue;
            }
        };
        parsed += 1;
        let mut names = Vec::new();
        collect(&ir, &mut names);
        let mut in_this_file: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for n in &names {
            let e = seen.entry(n.clone()).or_insert((0, 0));
            e.1 += 1;
            in_this_file.insert(n.clone());
        }
        for n in in_this_file {
            seen.get_mut(&n).unwrap().0 += 1;
        }
    }

    println!("# fidelity_scan: {parsed} fixtures parsed, {parse_fail} parse-failed");
    println!("# {} distinct op-kinds", seen.len());
    println!("{:<48} {:>8} {:>10}", "op-kind", "fixtures", "occurs");
    for (name, (nfix, occ)) in &seen {
        println!("{name:<48} {nfix:>8} {occ:>10}");
    }
}
