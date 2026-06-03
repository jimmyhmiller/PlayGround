//! Diff the union-find ranges vs the two-phase aliasing-ranges for one source.
//! Usage: cargo run -q --example ranges_diff -- path/to/fixture.js
use std::fs;

fn main() {
    let path = std::env::args().nth(1).expect("usage: ranges_diff <file.js>");
    let src = fs::read_to_string(&path).expect("read");
    let cfg = match jsir_ssa::compile_ssa(&src) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compile error: {e}");
            return;
        }
    };
    let al = jsir_ssa::aliasing_ranges::analyze(&cfg);
    println!("==== CFG ====\n{}", jsir_ssa::print::print(&cfg));
    println!("\n==== aliasing ranges ====\n{}", jsir_ssa::mutability::render(&cfg, &al));
    {
        use jsir_ssa::types::{build_builtin_shapes, build_default_globals, is_react_like_name};
        let is_c = is_react_like_name(cfg.fn_name.as_deref().unwrap_or(""));
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        let types = jsir_ssa::infer_types::infer(&cfg, &shapes, &globals, is_c, Default::default());
        let eff = jsir_ssa::infer_effects::infer(&cfg, &types, &shapes, is_c);
        println!("\n==== effects per instr ====");
        for ie in &eff.instrs {
            if !ie.effects.is_empty() {
                println!("  bb{} instr{} (res {:?}): {:?}", ie.block.0, ie.index, ie.result, ie.effects);
            }
        }
    }
    println!("\n==== mutations_by_instr ====");
    let mb = jsir_ssa::aliasing_ranges::mutations_by_instr(&cfg);
    let mut keys: Vec<_> = mb.keys().cloned().collect();
    keys.sort_by_key(|(b, i)| (b.0, *i));
    for k in keys {
        println!("  bb{} instr{} -> {:?}", k.0 .0, k.1, mb[&k]);
    }
}
