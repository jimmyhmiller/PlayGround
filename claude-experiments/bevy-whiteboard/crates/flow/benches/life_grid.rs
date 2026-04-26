//! Criterion benchmark for the rule-fire hot path. Uses the same
//! Game-of-Life templates the stress-test canvases run, programmatically
//! wires an N×N toroidal grid, and times advancing the sim through one
//! generation. The interesting cost is `try_fire` × (N² × 8 reports +
//! N² on_pulse + 1 clock) per generation.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use flow::{Expr, Sim, Value, dsl};

const LIFE_FLOW: &str = r#"
node LifeCell {
    slots {
        alive: Int = 0
        period_ns: Int = 200000000
        reports_seen: Int = 0
        live_neighbors: Int = 0
    }
    on_spawn {
        self -> self : period_ns
        inject tick(nil)
    }
    rule on_tick {
        on tick(_)
        do {
            reports_seen := 0
            live_neighbors := 0
            emit_each report(alive) to out_neighbors()
            emit tick(nil) to self
        }
    }
    rule on_report_partial {
        on report(v)
        when reports_seen + 1 < 8
        do {
            reports_seen := reports_seen + 1
            live_neighbors := live_neighbors + v
        }
    }
    rule on_report_final {
        on report(v)
        when reports_seen + 1 == 8
        do {
            reports_seen := reports_seen + 1
            live_neighbors := live_neighbors + v
            alive := if (alive == 1 && (live_neighbors == 2 || live_neighbors == 3)) || (alive == 0 && live_neighbors == 3) then 1 else 0
        }
    }
}
"#;

fn build_life_grid(w: usize, h: usize) -> Sim {
    let mut sim = Sim::new(1);
    // 60×60 produces ~32k firings/generation; bump well past the
    // default 10k safety cap so the bench scales without panics.
    sim.max_steps_per_instant = 10_000_000;
    dsl::register_classes(&mut sim, LIFE_FLOW).unwrap();

    let mut cells = vec![vec![flow::NodeId(0); w]; h];
    for y in 0..h {
        for x in 0..w {
            let name = format!("Cell_{}_{}", x, y);
            let id = sim.instantiate("LifeCell", &name).unwrap();
            // Deterministic initial fill (~20% alive) so the bench
            // exercises both alive/dead branches of the final rule.
            if let Some(node) = sim.nodes.get_mut(&id) {
                let alive = ((x * 7 + y * 13) % 5 == 0) as i64;
                node.slots.insert("alive".into(), Value::Int(alive));
            }
            cells[y][x] = id;
        }
    }

    let one_ms = Expr::int(1_000_000);

    let neighbors: [(isize, isize); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];
    for y in 0..h {
        for x in 0..w {
            for (dx, dy) in neighbors {
                let nx = ((x as isize + dx).rem_euclid(w as isize)) as usize;
                let ny = ((y as isize + dy).rem_euclid(h as isize)) as usize;
                sim.add_edge(cells[y][x], cells[ny][nx], one_ms.clone());
            }
        }
    }

    sim
}

/// Run a Sim through one full Life generation. Cells self-tick at T=0,
/// reports arrive at T=1ms, B3/S23 update happens immediately after.
/// 5ms is comfortably past the report-delivery window.
fn run_one_generation(sim: &mut Sim) {
    sim.run_until(sim.now_ns + 5_000_000);
}

fn bench_one_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("life_one_generation");
    // Throughput keyed on number of cells so output reads as
    // "elements/sec = cells processed per sec across all rule firings".
    for size in [10usize, 20, 30] {
        let cells = size * size;
        group.throughput(Throughput::Elements(cells as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // iter_batched_ref builds the sim outside the timed
            // section so setup cost (template lowering, edge wiring)
            // doesn't pollute the measurement.
            b.iter_batched_ref(
                || build_life_grid(size, size),
                |sim| {
                    run_one_generation(sim);
                    std::hint::black_box(sim.now_ns);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_one_generation);
criterion_main!(benches);
