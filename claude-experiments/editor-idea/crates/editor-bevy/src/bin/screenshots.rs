//! Screenshot suite: launches the editor, cycles through a fixed list
//! of scenarios (each is an initial document + a `prepare` transform
//! that places the caret / selection / scroll), takes one PNG per
//! scenario, then exits. Output goes to `target/screenshots/` at the
//! workspace root.
//!
//! Run with: `cargo run -p editor_bevy --bin screenshots`
//!
//! The app flashes a window briefly — this is not a headless run
//! because the GPU pipeline + window surface must be live for the
//! screenshot API to read pixels back.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PrimaryWindow;
use editor_bevy::{EditorPlugin, EditorRes, Scroll};
use editor_core::selection::{Range, Selection};
use editor_core::state::EditorState;
use editor_core::transaction::Transaction;
use ropey::Rope;

const WARMUP_FRAMES: u32 = 30;
const FRAMES_PER_SCENARIO: u32 = 10;
const SCREENSHOT_FRAME: u32 = 5;

struct Scenario {
    name: &'static str,
    initial: &'static str,
    /// Mutates state + scroll after the initial doc is installed.
    prepare: fn(&mut EditorState, &mut Scroll),
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "01_initial",
        initial: "fn main() {\n    println!(\"hello, editor\");\n}\n",
        prepare: |_, _| {},
    },
    Scenario {
        name: "02_caret_mid_line",
        initial: "fn main() {\n    println!(\"hello, editor\");\n}\n",
        prepare: |s, _| {
            // Caret in the middle of line 2 ("println!..."), after "    print"
            *s = s.apply(&Transaction::new().select(Selection::cursor(21)));
        },
    },
    Scenario {
        name: "03_end_of_file",
        initial: "fn main() {\n    println!(\"hello, editor\");\n}\n",
        prepare: |s, _| {
            let n = s.doc.len_chars();
            *s = s.apply(&Transaction::new().select(Selection::cursor(n)));
        },
    },
    Scenario {
        name: "04_selection_single_line",
        initial: "fn main() {\n    println!(\"hello, editor\");\n}\n",
        prepare: |s, _| {
            // Select `hello, editor` inside the string on line 2.
            // Line 2 starts at char 12; the string starts at offset 13 ("println!(\""
            // length 10 = chars 12..22), so 22..35 is "hello, editor".
            *s = s.apply(&Transaction::new().select(Selection::single(Range::new(22, 35))));
        },
    },
    Scenario {
        name: "05_selection_multi_line",
        initial: "alpha\nbravo\ncharlie\ndelta\n",
        prepare: |s, _| {
            // Select from the 'r' in "bravo" (char 8) through the 'l' in
            // "delta" (past "charlie\nd", which is char 21).
            *s = s.apply(&Transaction::new().select(Selection::single(Range::new(8, 21))));
        },
    },
    Scenario {
        name: "06_empty_doc",
        initial: "",
        prepare: |_, _| {},
    },
    Scenario {
        name: "07_utf8_and_cjk",
        initial: "hello, 世界 🦀\ncafé résumé\nαβγδε\n",
        prepare: |s, _| {
            // Caret past the rust crab — exercises multi-byte glyph
            // placement.
            *s = s.apply(&Transaction::new().select(Selection::cursor(11)));
        },
    },
    Scenario {
        name: "08_scrolled_long_doc",
        initial: "",
        prepare: |s, scroll| {
            // Override the doc: 80 numbered lines. Simpler here than
            // spelling out an 80-line literal above.
            let mut text = String::new();
            for i in 0..80 {
                text.push_str(&format!("line {:>3}  the quick brown fox jumps over the lazy dog\n", i));
            }
            *s = EditorState::new(Rope::from_str(&text), Selection::cursor(0))
                .with_indent_unit("    ");
            // Scroll to roughly line 30 (30 * LINE_HEIGHT).
            scroll.0 = 30.0 * editor_bevy::LINE_HEIGHT;
        },
    },
    Scenario {
        name: "09_indented_block_with_tabs",
        initial: "if condition {\n\tfoo();\n\tbar();\n}\n",
        prepare: |s, _| {
            // Caret at start of tabbed line — visually asserts the tab's
            // advance width matches the selection / caret math.
            *s = s.apply(&Transaction::new().select(Selection::cursor(16)));
        },
    },
    Scenario {
        // Regression probe: text, then 10 empty lines, then more text.
        // Caret is placed at the START of the second text block (char 17,
        // i.e. immediately before "AFTER"). If empty lines contribute the
        // correct LINE_HEIGHT, the caret will sit directly to the left of
        // the "A" in "AFTER". If empty lines don't add height, the caret
        // will appear above "AFTER" — visibly misaligned.
        name: "10_empty_lines_between_text",
        initial: "BEFORE\n\n\n\n\n\n\n\n\n\nAFTER\n",
        prepare: |s, _| {
            // "BEFORE" = 6 chars + "\n" = 7, + 10 newlines = char 17.
            *s = s.apply(&Transaction::new().select(Selection::cursor(17)));
        },
    },
];

#[derive(Resource)]
struct Runner {
    index: usize,
    frame_in_scenario: u32,
    warmup_remaining: u32,
    output_dir: PathBuf,
    applied_this_scenario: bool,
}

fn main() {
    let output_dir = workspace_root().join("target/screenshots");
    std::fs::create_dir_all(&output_dir).expect("mkdir target/screenshots");
    eprintln!("writing screenshots to {}", output_dir.display());

    let mut app = App::new();
    app.insert_resource(EditorRes(
        EditorState::new(Rope::from_str(SCENARIOS[0].initial), Selection::cursor(0))
            .with_indent_unit("    "),
    ));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "editor screenshots".into(),
            // Pin resolution for reproducibility.
            resolution: (900u32, 600u32).into(),
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(EditorPlugin);
    app.insert_resource(Runner {
        index: 0,
        frame_in_scenario: 0,
        warmup_remaining: WARMUP_FRAMES,
        output_dir,
        applied_this_scenario: false,
    });
    app.add_systems(Update, drive_scenarios);
    app.run();
}

fn drive_scenarios(
    mut runner: ResMut<Runner>,
    mut state: ResMut<EditorRes>,
    mut scroll: ResMut<Scroll>,
    mut commands: Commands,
    primary: Query<Entity, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
) {
    if runner.warmup_remaining > 0 {
        runner.warmup_remaining -= 1;
        return;
    }

    if runner.index >= SCENARIOS.len() {
        exit.write(AppExit::Success);
        return;
    }

    let scenario = &SCENARIOS[runner.index];

    // Frame 0 of the scenario: install its initial doc + prepare transform.
    if !runner.applied_this_scenario {
        state.0 = EditorState::new(Rope::from_str(scenario.initial), Selection::cursor(0))
            .with_indent_unit("    ");
        scroll.0 = 0.0;
        (scenario.prepare)(&mut state.0, &mut scroll);
        runner.applied_this_scenario = true;
        runner.frame_in_scenario = 0;
        return;
    }

    runner.frame_in_scenario += 1;

    if runner.frame_in_scenario == SCREENSHOT_FRAME {
        // Layout has had a few frames to settle. Capture.
        if primary.single().is_ok() {
            let path = runner.output_dir.join(format!("{}.png", scenario.name));
            eprintln!("  → {}", path.display());
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
        }
    }


    if runner.frame_in_scenario >= FRAMES_PER_SCENARIO {
        runner.index += 1;
        runner.applied_this_scenario = false;
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `crates/editor-bevy`; workspace root is two up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("."))
}
