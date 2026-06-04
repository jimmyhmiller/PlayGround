//! Probe for the in-place-memoization prototype: dump the JSIR statement tree
//! (with node_ids), the CFG (with per-instruction SrcRef → originating
//! statement), and the analysis scopes (deps/outputs and the statement each
//! output value comes from). Usage: inplace_probe [file.js]
use std::collections::HashMap;
use jsir_ssa::cfg::Value;

fn dump_ir(op: &jsir_ir::Op, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    let nid = op.node_id.map(|n| format!(" #{n}")).unwrap_or_default();
    out.push_str(&format!("{pad}{}{nid}\n", op.name));
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                dump_ir(o, depth + 1, out);
            }
        }
    }
}

fn main() {
    let src = match std::env::args().nth(1) {
        Some(p) => std::fs::read_to_string(p).unwrap(),
        None => "function Component(props){ let i = 0; while (i < 3) { i = i + 1; } const obj = { a: props.x }; return obj; }".to_string(),
    };
    println!("=== SOURCE ===\n{src}\n");

    let ir = jsir_swc::source_to_ir(&src).expect("source_to_ir");
    let mut s = String::new();
    dump_ir(&ir, 0, &mut s);
    println!("=== JSIR TREE (name #node_id) ===\n{s}");

    // Lower + analyze (mirror codegen::compile's analysis half).
    let mut cfg = jsir_ssa::lower::lower_function(&ir).expect("lower");
    jsir_ssa::ssa::construct(&mut cfg);
    jsir_ssa::constfold::fold_constants(&mut cfg);
    let r = jsir_ssa::aliasing_ranges::analyze(&cfg);
    let infos = jsir_ssa::scopes::analyze(&cfg, &r);

    println!("=== CFG ===\n{}", jsir_ssa::print::print(&cfg));

    // value -> originating statement node_id (via SrcRef on its defining instr).
    let mut val_src: HashMap<Value, u32> = HashMap::new();
    println!("\n=== per-instruction SrcRef ===");
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(res), Some(sr)) = (ins.result, ins.src) {
                val_src.insert(res, sr.stmt_node_id);
                println!("  %{} <- stmt #{} expr #{:?}", res.0, sr.stmt_node_id, sr.expr_node_id);
            }
        }
    }

    println!("\n=== SCOPES (after analysis) ===");
    for (i, info) in infos.iter().enumerate() {
        let deps: Vec<String> = info.deps.iter().map(|v| format!("%{}", v.0)).collect();
        let outs: Vec<String> = info.outputs.iter().map(|v| format!("%{}", v.0)).collect();
        let out_src: Vec<String> = info.outputs.iter().map(|v| {
            val_src.get(v).map(|n| format!("%{}@stmt#{n}", v.0)).unwrap_or(format!("%{}@?", v.0))
        }).collect();
        let dep_src: Vec<String> = info.deps.iter().map(|v| {
            val_src.get(v).map(|n| format!("%{}@stmt#{n}", v.0)).unwrap_or(format!("%{}@?", v.0))
        }).collect();
        println!("  scope[{i}] mutable={} start={} end={} values={:?}",
            info.scope.mutable, info.scope.start, info.scope.end,
            info.scope.values.iter().map(|v| v.0).collect::<Vec<_>>());
        println!("        deps={deps:?}  outputs={outs:?}");
        println!("        deps@stmt={dep_src:?}  outputs@stmt={out_src:?}");
    }

    println!("\n=== IN-PLACE MEMOIZED OUTPUT (loop kept verbatim) ===");
    match jsir_ssa::memoize_plan::memoize_inplace(&cfg, &infos, &r, &ir) {
        Ok(memo) => match jsir_swc::ir_to_source(&memo) {
            Ok(out) => println!("{out}"),
            Err(e) => println!("ir_to_source error: {e}"),
        },
        Err(e) => println!("memoize_inplace bailed: {e}"),
    }
}
