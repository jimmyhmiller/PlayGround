//! A long-running program with a varied, churning workload and the memscope
//! agent attached, so you can drive the `memscope` CLI against it live:
//!
//!   cargo run -p demo --release --bin serve        # in one terminal
//!   cargo run -p memscope-cli --release -- monitor # in another
//!
//! The workload allocates several distinct types and keeps a fluctuating live
//! set so the monitor shows real, moving numbers.

use std::collections::HashMap;

use memscope::{Mode, MemScope};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Particle {
    pos: [f64; 3],
    vel: [f64; 3],
    mass: f64,
    id: u64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Session {
    user: String,
    tokens: Vec<u32>,
    active: bool,
}

fn main() {
    memscope::set_mode(Mode::Full);
    // MEMSCOPE_RECORD=<path> streams every allocation to a self-contained file.
    if let Ok(path) = std::env::var("MEMSCOPE_RECORD") {
        memscope::record_to_file(&path).expect("failed to start recording");
    }
    let sock = memscope::start_agent().expect("agent failed to start");
    println!("serve: workload running. Attach with:");
    println!("    cargo run -p memscope-cli --release -- monitor --sock {sock}");
    println!("    cargo run -p memscope-cli --release -- dump    --sock {sock} --out /tmp/heap.json");

    // Several pools with different lifetimes so the live heap is interesting.
    let mut particles: Vec<Box<Particle>> = Vec::new();
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    let mut tick: u64 = 0;
    loop {
        tick += 1;

        // Grow particles, occasionally trim — a long-lived, growing pool.
        for i in 0..200 {
            particles.push(Box::new(Particle {
                pos: [i as f64, tick as f64, 0.0],
                vel: [0.1, 0.0, -0.1],
                mass: 1.0,
                id: tick * 1000 + i,
            }));
        }
        if particles.len() > 20_000 {
            particles.drain(0..5_000);
        }

        // Sessions: insert and expire — churning HashMap.
        for i in 0..100 {
            let id = tick * 100 + i;
            sessions.insert(
                id,
                Session {
                    user: format!("user-{}", id % 997),
                    tokens: (0..(id % 16) as u32).collect(),
                    active: id % 3 == 0,
                },
            );
        }
        sessions.retain(|&k, _| k + 3000 > tick * 100);

        // Short-lived byte buffers — transient allocations (freed next tick).
        buffers.clear();
        for _ in 0..50 {
            buffers.push(vec![0u8; 1024 + (tick as usize % 4096)]);
        }

        // A bounded ring of strings.
        labels.push(format!("event-{tick}-{}", "x".repeat((tick % 40) as usize)));
        if labels.len() > 500 {
            labels.remove(0);
        }

        std::hint::black_box((&particles, &sessions, &buffers, &labels));
        std::thread::sleep(std::time::Duration::from_millis(50));

        if tick % 100 == 0 {
            let s = memscope::stats();
            println!(
                "serve: tick {tick}  live={} KiB  particles={}  sessions={}",
                s.live_bytes / 1024,
                particles.len(),
                sessions.len()
            );
        }
    }
}
