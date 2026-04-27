//! Diagnostic: verify every sim node in a life canvas has a position
//! entry in its `visual.json`. Stray nodes would land at a default
//! grid position (a column to the left of the main grid) — that's
//! the "lone disconnected cell" symptom we're chasing.

use std::path::Path;

#[test]
fn life_canvases_have_no_stray_nodes() {
    for ex in &[
        "life_5x5_blinker",
        "life_30x30_random",
        "life_50x50_random",
    ] {
        let path = format!(
            "{}/examples/{}.whiteboard",
            env!("CARGO_MANIFEST_DIR").trim_end_matches("/crates/flow-bevy"),
            ex
        );
        let canvas = flow_bevy::canvas::load_canvas(Path::new(&path), 1).unwrap();

        let visual_keys: std::collections::HashSet<String> =
            canvas.visual.nodes.keys().cloned().collect();
        let sim_names: Vec<String> = canvas.sim.nodes.values().map(|n| n.name.clone()).collect();
        let stray: Vec<&String> = sim_names.iter().filter(|n| !visual_keys.contains(*n)).collect();
        let visual_only: Vec<String> = visual_keys
            .iter()
            .filter(|k| !sim_names.contains(*k))
            .cloned()
            .collect();
        println!(
            "{}: sim={} visual={} stray={:?} visual_only={:?}",
            ex,
            sim_names.len(),
            visual_keys.len(),
            stray,
            visual_only
        );
        assert!(
            stray.is_empty(),
            "{}: sim has nodes not in visual.json: {:?}",
            ex,
            stray
        );
    }
}
