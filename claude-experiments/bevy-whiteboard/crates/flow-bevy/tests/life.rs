//! End-to-end Game of Life test. Loads the generated 5×5 blinker
//! whiteboard, runs one generation, and asserts the pattern flipped
//! from a horizontal row to a vertical column. Validates the
//! synchronous-tick rule design (8-report accumulator + B3/S23) under
//! the canvas pipeline.

use std::path::PathBuf;

use flow::{Sim, Value};
use flow_bevy::canvas::load_canvas;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn alive(sim: &Sim, name: &str) -> i64 {
    let n = sim
        .nodes
        .values()
        .find(|n| n.name == name)
        .unwrap_or_else(|| panic!("no node named `{}`", name));
    match n.slots.get("alive") {
        Some(Value::Int(i)) => *i,
        other => panic!("{}.alive: expected Int, got {:?}", name, other),
    }
}

fn assert_grid(sim: &Sim, w: usize, h: usize, expected: &[(usize, usize)]) {
    let mut wrong = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let want = expected.contains(&(x, y));
            let got = alive(sim, &format!("Cell_{}_{}", x, y)) == 1;
            if want != got {
                wrong.push((x, y, want, got));
            }
        }
    }
    if !wrong.is_empty() {
        let mut msg = String::from("grid mismatch:\n");
        for (x, y, want, got) in &wrong {
            msg.push_str(&format!(
                "  ({},{}): expected {}, got {}\n",
                x, y, want, got
            ));
        }
        msg.push_str("\nactual grid:\n");
        for y in 0..h {
            for x in 0..w {
                msg.push(if alive(sim, &format!("Cell_{}_{}", x, y)) == 1 {
                    '#'
                } else {
                    '.'
                });
            }
            msg.push('\n');
        }
        panic!("{}", msg);
    }
}

#[test]
fn random_30x30_runs_many_generations() {
    let path = project_root().join("examples/life_30x30_random.whiteboard");
    let mut canvas = load_canvas(&path, 1).expect("load life_30x30_random.whiteboard");

    // Run ~10 generations (period 150ms × 12 with margin). Just verifying
    // the engine handles 30×30 without runtime errors and within the
    // max_steps_per_instant budget.
    canvas.sim.run_until(canvas.sim.now_ns + 1_800_000_000);

    let errs: Vec<_> = canvas
        .sim
        .error_counts
        .iter()
        .filter(|(_, c)| **c > 0)
        .collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}

#[test]
fn blinker_5x5_oscillates() {
    let path = project_root().join("examples/life_5x5_blinker.whiteboard");
    let mut canvas = load_canvas(&path, 1).expect("load life_5x5_blinker.whiteboard");

    // Initial: horizontal row centered at y=2.
    assert_grid(&canvas.sim, 5, 5, &[(1, 2), (2, 2), (3, 2)]);

    // Gen 1 completes at ~T=2ms (clock 1ms + cell 1ms + processing).
    // Run to T=100ms — well clear of the next pulse at T=201ms.
    canvas.sim.run_until(canvas.sim.now_ns + 100_000_000);
    assert_grid(&canvas.sim, 5, 5, &[(2, 1), (2, 2), (2, 3)]);

    // Gen 2 completes at ~T=202ms. Run to T=300ms.
    canvas.sim.run_until(canvas.sim.now_ns + 200_000_000);
    assert_grid(&canvas.sim, 5, 5, &[(1, 2), (2, 2), (3, 2)]);

    // Gen 3 completes at ~T=402ms. Run to T=500ms.
    canvas.sim.run_until(canvas.sim.now_ns + 200_000_000);
    assert_grid(&canvas.sim, 5, 5, &[(2, 1), (2, 2), (2, 3)]);

    // No runtime errors.
    let errs: Vec<_> = canvas
        .sim
        .error_counts
        .iter()
        .filter(|(_, c)| **c > 0)
        .collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}
