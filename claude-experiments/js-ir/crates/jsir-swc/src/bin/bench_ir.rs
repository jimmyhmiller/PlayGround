//! Benchmark the two IR storage layouts on the supported scope: build + print.
//!
//! Compares the AoS `Op` tree against the SoA `Module` on (1) resident memory
//! footprint — bytes and number of distinct heap allocations — and (2) print
//! throughput, over a large generated program.
//!
//!   cargo run --release -p jsir-swc --bin bench_ir
//!
//! Methodology: footprint sums container capacities for both layouts and
//! excludes attribute *payload* internals (identical content in both — a wash),
//! so the delta reflects the layout itself.

use std::time::Instant;

use jsir_ir::{Attr, Block, Module, Op, Region, ValueId};

/// Resident `(bytes, blocks)` of an AoS `Op` tree, same accounting as
/// `Module::footprint`. The root struct is standalone; every other op's struct
/// bytes live inside its parent block's `Vec<Op>` buffer.
fn aos_footprint(root: &Op) -> (usize, usize) {
    let mut bytes = std::mem::size_of::<Op>(); // the standalone root struct
    let mut blocks = 0usize;
    visit(root, &mut bytes, &mut blocks);
    (bytes, blocks)
}

fn visit(op: &Op, bytes: &mut usize, blocks: &mut usize) {
    let buf = |cap: usize, elem: usize, bytes: &mut usize, blocks: &mut usize| {
        if cap > 0 {
            *bytes += cap * elem;
            *blocks += 1;
        }
    };
    if op.name.capacity() > 0 {
        *bytes += op.name.capacity();
        *blocks += 1;
    }
    buf(op.operands.capacity(), std::mem::size_of::<ValueId>(), bytes, blocks);
    buf(op.results.capacity(), std::mem::size_of::<ValueId>(), bytes, blocks);
    buf(op.attrs.capacity(), std::mem::size_of::<(String, Attr)>(), bytes, blocks);
    buf(op.successors.capacity(), std::mem::size_of::<jsir_ir::Successor>(), bytes, blocks);
    for s in &op.successors {
        buf(s.args.capacity(), std::mem::size_of::<ValueId>(), bytes, blocks);
    }
    buf(op.regions.capacity(), std::mem::size_of::<Region>(), bytes, blocks);
    for r in &op.regions {
        visit_region(r, bytes, blocks);
    }
}

fn visit_region(region: &Region, bytes: &mut usize, blocks: &mut usize) {
    if region.blocks.capacity() > 0 {
        *bytes += region.blocks.capacity() * std::mem::size_of::<Block>();
        *blocks += 1;
    }
    for b in &region.blocks {
        if b.args.capacity() > 0 {
            *bytes += b.args.capacity() * std::mem::size_of::<ValueId>();
            *blocks += 1;
        }
        if b.ops.capacity() > 0 {
            *bytes += b.ops.capacity() * std::mem::size_of::<Op>(); // op structs live here
            *blocks += 1;
        }
        for o in &b.ops {
            visit(o, bytes, blocks);
        }
    }
}

/// A large, representative program: var-decls with nested binary expressions,
/// functions (region nesting), and if/else, so both the AST dialect's leaf ops
/// and its regions are exercised.
fn generate(stmts: usize) -> String {
    let mut s = String::with_capacity(stmts * 48);
    for i in 0..stmts {
        match i % 4 {
            0 => s.push_str(&format!("var v{i} = ({i} + {a}) * 3 - {b} / 2;\n", a = i * 2, b = i + 1)),
            1 => s.push_str(&format!(
                "function f{i}(x, y) {{ return x * {i} + y - {a}; }}\n",
                a = i % 7
            )),
            2 => s.push_str(&format!(
                "if (v{j} > {i}) {{ v{j} = v{j} + 1; }} else {{ v{j} = {i}; }}\n",
                j = i.saturating_sub(2)
            )),
            _ => s.push_str(&format!("var s{i} = \"item-\" + {i} + \"-end\";\n")),
        }
    }
    s
}

fn count_ops(op: &Op) -> usize {
    let mut n = 1;
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                n += count_ops(o);
            }
        }
    }
    n
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MiB", b as f64 / (1 << 20) as f64)
    } else {
        format!("{:.1} KiB", b as f64 / 1024.0)
    }
}

fn main() {
    let stmts: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let src = generate(stmts);
    eprintln!("generated {} statements ({} bytes of JS)\n", stmts, src.len());

    // ── build ──
    let t = Instant::now();
    let op = jsir_swc::source_to_ir(&src).expect("source_to_ir");
    let build_aos = t.elapsed();

    let t = Instant::now();
    let module = Module::from_op(&op);
    let build_soa = t.elapsed();

    let ops = count_ops(&op);

    // ── footprint ──
    let (aos_bytes, aos_blocks) = aos_footprint(&op);
    let (soa_bytes, soa_blocks) = module.footprint();

    // ── print (warm up, then average) ──
    let iters = 20;
    let mut sink = 0usize;
    let _ = op.print();
    let t = Instant::now();
    for _ in 0..iters {
        sink ^= op.print().len();
    }
    let print_aos = t.elapsed() / iters;

    let _ = module.print();
    let t = Instant::now();
    for _ in 0..iters {
        sink ^= module.print().len();
    }
    let print_soa = t.elapsed() / iters;
    std::hint::black_box(sink);

    let out_len = op.print().len();
    let mbps = |d: std::time::Duration| (out_len as f64 / (1 << 20) as f64) / d.as_secs_f64();

    // ── report ──
    println!("ops: {ops}   printed text: {}", fmt_bytes(out_len));
    println!("sizeof(Op) = {} B   SoA hot bytes/op = {} B\n",
        std::mem::size_of::<Op>(), Module::per_op_hot_bytes());

    println!("{:<14} {:>14} {:>14}   {:>8}", "", "AoS Op tree", "SoA Module", "ratio");
    println!("{:<14} {:>14} {:>14}   {:>7.2}x",
        "footprint", fmt_bytes(aos_bytes), fmt_bytes(soa_bytes),
        aos_bytes as f64 / soa_bytes as f64);
    println!("{:<14} {:>14} {:>14}   {:>7.2}x",
        "heap blocks", aos_blocks, soa_blocks,
        aos_blocks as f64 / soa_blocks as f64);
    println!("{:<14} {:>13.1} {:>13.1}   {:>7.2}x",
        "blocks/op", aos_blocks as f64 / ops as f64, soa_blocks as f64 / ops as f64,
        (aos_blocks as f64) / (soa_blocks as f64));
    println!();
    println!("{:<14} {:>14} {:>14}   {:>8}", "", "AoS", "SoA", "speedup");
    println!("{:<14} {:>14?} {:>14?}", "build", build_aos, build_soa);
    println!("  (AoS build = swc parse + lower; SoA build = lower the Op tree)");
    println!("{:<14} {:>11.2?}/it {:>11.2?}/it   {:>7.2}x   ({:.0} vs {:.0} MiB/s)",
        "print", print_aos, print_soa,
        print_aos.as_secs_f64() / print_soa.as_secs_f64(),
        mbps(print_aos), mbps(print_soa));
}
