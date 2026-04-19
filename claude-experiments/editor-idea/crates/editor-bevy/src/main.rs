use std::fs;
use std::path::PathBuf;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use editor_bevy::build_app;

const DEFAULT_DOC: &str = "fn main() {\n    println!(\"hello, editor\");\n}\n";

fn main() {
    let args = Args::parse();
    let initial = match args.source {
        Source::Default => DEFAULT_DOC.to_string(),
        Source::Synthetic(n) => synthetic_doc(n),
        Source::File(path) => match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("failed to read {}: {err}", path.display());
                std::process::exit(1);
            }
        },
    };

    let lines = initial.matches('\n').count().saturating_add(1);
    eprintln!(
        "opening doc: {} bytes, {} lines",
        initial.len(),
        lines,
    );

    let mut app = build_app(&initial);
    if args.log_frames {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Update, log_frame_time);
    }
    app.run();
}

enum Source {
    Default,
    Synthetic(usize),
    File(PathBuf),
}

struct Args {
    source: Source,
    log_frames: bool,
}

impl Args {
    fn parse() -> Self {
        // Tiny hand-rolled parser; keeps the bin free of clap and co.
        let mut source = Source::Default;
        let mut log_frames = false;
        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--file" => {
                    let Some(p) = args.next() else {
                        fatal("--file needs a path");
                    };
                    source = Source::File(PathBuf::from(p));
                }
                "--lines" => {
                    let Some(n) = args.next().and_then(|s| s.parse().ok()) else {
                        fatal("--lines needs an integer");
                    };
                    source = Source::Synthetic(n);
                }
                "--log-frames" => log_frames = true,
                "-h" | "--help" => {
                    eprintln!(
                        "editor [--file PATH | --lines N] [--log-frames]\n\
                         \n\
                         Options:\n\
                         --file PATH    load a file into the editor\n\
                         --lines N      generate N synthetic lines for benchmarking\n\
                         --log-frames   print rolling frame time to stderr every 60 frames"
                    );
                    std::process::exit(0);
                }
                other => fatal(&format!("unknown arg: {other}")),
            }
        }
        Self { source, log_frames }
    }
}

fn fatal(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// Build a doc with `n` lines of the form `line   12  the quick brown fox ...`
/// so we have realistic glyph counts per line rather than one-char lines.
fn synthetic_doc(n: usize) -> String {
    let mut s = String::with_capacity(n * 60);
    for i in 0..n {
        s.push_str(&format!(
            "line {:>6}  the quick brown fox jumps over the lazy dog\n",
            i
        ));
    }
    s
}

/// Every 60 frames, log the rolling-average frame time. Enough to eyeball
/// where large documents start costing real milliseconds without flooding
/// stderr.
fn log_frame_time(
    diagnostics: Res<DiagnosticsStore>,
    mut frame_count: Local<u32>,
) {
    *frame_count += 1;
    if *frame_count % 60 != 0 {
        return;
    }
    let avg_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.average())
        .unwrap_or(0.0);
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.average())
        .unwrap_or(0.0);
    eprintln!("frame {:>5}: {:6.2} ms ({:5.1} fps)", *frame_count, avg_ms, fps);
}
