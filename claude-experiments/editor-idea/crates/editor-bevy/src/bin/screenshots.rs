//! Screenshot suite: launches the editor, cycles through a fixed list
//! of scenarios (each is an initial document + a `prepare` transform
//! that places the caret / selection / scroll), takes one PNG per
//! scenario, then exits. Output goes to `target/screenshots/` at the
//! workspace root.
//!
//! Run with: `cargo run -p editor_bevy --bin screenshots`

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PrimaryWindow;
use editor_bevy::{
    build_app, setup_camera_and_font, spawn_editor, EditorRect, EditorScroll, EditorStateComp,
    FocusedEditor, MonoFont,
};
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
    prepare: fn(&mut EditorState, &mut EditorScroll),
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
            *s = s.apply(&Transaction::new().select(Selection::single(Range::new(22, 35))));
        },
    },
    Scenario {
        name: "05_selection_multi_line",
        initial: "alpha\nbravo\ncharlie\ndelta\n",
        prepare: |s, _| {
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
            *s = s.apply(&Transaction::new().select(Selection::cursor(11)));
        },
    },
    Scenario {
        name: "08_scrolled_long_doc",
        initial: "",
        prepare: |s, scroll| {
            let mut text = String::new();
            for i in 0..80 {
                text.push_str(&format!(
                    "line {:>3}  the quick brown fox jumps over the lazy dog\n",
                    i
                ));
            }
            *s = EditorState::new(Rope::from_str(&text), Selection::cursor(0))
                .with_indent_unit("    ");
            scroll.y = 30.0 * editor_bevy::LINE_HEIGHT;
        },
    },
    Scenario {
        name: "09_indented_block_with_tabs",
        initial: "if condition {\n\tfoo();\n\tbar();\n}\n",
        prepare: |s, _| {
            *s = s.apply(&Transaction::new().select(Selection::cursor(16)));
        },
    },
    Scenario {
        name: "11_caret_after_indent_empty_line",
        initial: "fn main() {\n    println!(\"hello, editor\");\n    \n}\n",
        prepare: |s, _| {
            *s = s.apply(&Transaction::new().select(Selection::cursor(47)));
        },
    },
    Scenario {
        name: "10_empty_lines_between_text",
        initial: "BEFORE\n\n\n\n\n\n\n\n\n\nAFTER\n",
        prepare: |s, _| {
            *s = s.apply(&Transaction::new().select(Selection::cursor(17)));
        },
    },
    Scenario {
        // Visual check only — verifies two overlapping editors render
        // and the topmost one draws on top via z-order. The second
        // editor is spawned lazily in a Startup-after system, so for
        // this scenario the first editor still gets the `prepare`
        // applied (it's the one this suite mutates). The second editor
        // renders its own hardcoded demo doc.
        name: "12_two_editors",
        initial: "fn main() {\n    println!(\"hello, editor\");\n}\n",
        prepare: |_, _| {},
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

    let mut app = build_app(SCENARIOS[0].initial);
    // Spawn a second editor so the last scenario can visually check
    // that two editors render with correct z-ordering.
    app.add_systems(
        Startup,
        (|mut commands: Commands, font: Res<MonoFont>| {
            spawn_editor(
                &mut commands,
                &font,
                "// second editor\nfn fib(n: u32) -> u32 {\n    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n}\n",
                EditorRect {
                    pos: Vec2::new(160.0, 180.0),
                    size: Vec2::new(520.0, 320.0),
                    z: 2.0,
                },
            );
        })
            .after(setup_camera_and_font),
    );
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
    mut commands: Commands,
    primary: Query<Entity, With<PrimaryWindow>>,
    focused: Res<FocusedEditor>,
    mut editors: Query<(&mut EditorStateComp, &mut EditorScroll)>,
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

    if !runner.applied_this_scenario {
        let Some(target) = focused.0 else {
            return;
        };
        let Ok((mut state, mut scroll)) = editors.get_mut(target) else {
            return;
        };
        state.0 = EditorState::new(Rope::from_str(scenario.initial), Selection::cursor(0))
            .with_indent_unit("    ");
        *scroll = EditorScroll::default();
        (scenario.prepare)(&mut state.0, &mut scroll);
        runner.applied_this_scenario = true;
        runner.frame_in_scenario = 0;
        return;
    }

    runner.frame_in_scenario += 1;

    if runner.frame_in_scenario == SCREENSHOT_FRAME {
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("."))
}
