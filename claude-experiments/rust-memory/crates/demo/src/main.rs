//! End-to-end smoke test of `memscope-core`: install the tracking allocator,
//! exercise Full and Sampled modes, and print the live heap + sites + stats.

use memscope_core::{self as mem, Mode};

#[global_allocator]
static GLOBAL: mem::MemScope = mem::MemScope::system();

#[derive(Debug)]
#[allow(dead_code)]
struct Widget {
    id: u64,
    name: String,
    coords: [f64; 3],
}

fn workload(retain: &mut Vec<Box<Widget>>) {
    // Boxes we keep alive (stay in the live set).
    for i in 0..1000u64 {
        retain.push(Box::new(Widget {
            id: i,
            name: format!("widget-{i}"),
            coords: [i as f64, 0.0, 0.0],
        }));
    }
    // Transient vectors that get freed before the snapshot.
    for _ in 0..500 {
        let v: Vec<u64> = (0..64).collect();
        std::hint::black_box(&v);
    }
}

fn main() {
    println!("== memscope-core demo ==\n");

    // --- Full mode: exact tracking ---
    mem::set_mode(Mode::Full);
    let mut retained: Vec<Box<Widget>> = Vec::new();
    workload(&mut retained);

    let s = mem::stats();
    println!("[Full] after workload:");
    println!("  mode               = {:?}", s.mode);
    println!("  total allocations  = {}", s.total_allocs);
    println!("  total bytes allocd = {}", s.total_alloc_bytes);
    println!("  live bytes         = {}", s.live_bytes);
    println!("  dropped events     = {}", s.dropped_events);

    let dump = mem::snapshot();
    println!("\n[Full] heap snapshot:");
    println!("  live allocations   = {}", dump.live.len());
    println!("  total live bytes   = {}", dump.total_live_bytes);
    println!("  distinct sites     = {}", dump.sites.len());
    println!("  sample scale       = {}", dump.sample_scale);

    // Aggregate live bytes by site (the raw, pre-symbolication view).
    let mut by_site: std::collections::HashMap<u32, (u64, u64)> = std::collections::HashMap::new();
    for l in &dump.live {
        if l.site.is_some() {
            let e = by_site.entry(l.site.0).or_default();
            e.0 += 1;
            e.1 += l.size;
        }
    }
    let mut ranked: Vec<_> = by_site.into_iter().collect();
    ranked.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    println!("\n  top sites by live bytes (site id : count, bytes, depth):");
    for (site, (count, bytes)) in ranked.iter().take(5) {
        let depth = dump
            .sites
            .iter()
            .find(|s| s.id == *site)
            .map(|s| s.frames.len())
            .unwrap_or(0);
        println!("    site {site:<4} : {count:>5} allocs, {bytes:>8} bytes, {depth} frames");
    }

    // Drain a few events from the stream to prove the ring works.
    let mut evs = Vec::new();
    let n = mem::drain_events(&mut evs, 5);
    println!("\n  drained {n} live events (first few):");
    for ev in evs.iter().take(5) {
        println!(
            "    seq={:<6} {:?} addr={:#x} size={} site={:?}",
            ev.seq, ev.kind, ev.addr, ev.size, ev.site
        );
    }

    // Free everything; live set should collapse.
    retained.clear();
    let dump2 = mem::snapshot();
    println!(
        "\n[Full] after dropping retained: live allocs = {}, live bytes = {}",
        dump2.live.len(),
        dump2.total_live_bytes
    );

    // --- Sampled mode ---
    mem::set_mode(Mode::Sampled);
    mem::set_sample_rate(100);
    let mut keep: Vec<Vec<u8>> = Vec::new();
    for i in 0..100_000usize {
        keep.push(vec![0u8; 16]);
        if i % 7 == 0 {
            keep.pop();
        }
    }
    let ss = mem::stats();
    let sdump = mem::snapshot();
    println!("\n[Sampled rate=100] after 100k allocations:");
    println!("  sampled allocations recorded = {}", ss.total_allocs);
    println!("  live tracked allocations     = {}", sdump.live.len());
    println!("  sample scale                 = {}", sdump.sample_scale);
    println!(
        "  estimated live allocations   = {:.0}",
        sdump.live.len() as f64 * sdump.sample_scale
    );
    std::hint::black_box(&keep);

    println!("\n== done ==");
}
