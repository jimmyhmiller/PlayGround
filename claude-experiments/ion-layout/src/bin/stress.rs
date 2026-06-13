// Stress: deep chains (recursion depth), wide layers, and big CFGs; prints timing.
use ion_layout::core::*;
use std::time::Instant;

fn main() {
    // deep chain
    for n in [1000usize, 10000, 50000] {
        let nodes: Vec<NodeSpec> = (0..n).map(|_| NodeSpec { width: 100.0, height: 40.0, ..Default::default() }).collect();
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (i - 1, i)).collect();
        let t = Instant::now();
        let r = layout(&nodes, &edges);
        println!("chain {n}: {:?} (h={})", t.elapsed(), r.height as u64);
    }
    // big random CFG
    let mut st = 7u64;
    let mut rng = move || { st = (st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) >> 1; st };
    for n in [1000usize, 5000] {
        let nodes: Vec<NodeSpec> = (0..n).map(|_| NodeSpec { width: 120.0, height: 50.0, ..Default::default() }).collect();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n - 1 {
            edges.push((i, i + 1));
            if rng() % 3 == 0 { edges.push((i, (i + 2 + (rng() as usize % 20)).min(n - 1))); }
            if rng() % 11 == 0 { edges.push((i, i.saturating_sub(rng() as usize % 30))); }
        }
        edges.retain(|&(a, b)| a != b);
        let t = Instant::now();
        let r = layout(&nodes, &edges);
        println!("cfg {n} ({} edges): {:?} (w={} h={})", edges.len(), t.elapsed(), r.width as u64, r.height as u64);
    }
}
