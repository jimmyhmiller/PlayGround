//! CLI front-end over `inspector` views.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use ir::{load_manifest_file, manifest::Manifest};
use inspector::{handler_card, program_map, state_cell, validate_report};

#[derive(Parser)]
#[command(name = "inspector")]
#[command(about = "Inspect constrained-language IR manifests")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show the program map: events, state, effects, handlers.
    Show {
        /// Path to the manifest JSON file.
        manifest: PathBuf,
    },
    /// Show one handler's full footprint and body pointer.
    Handler {
        manifest: PathBuf,
        handler: String,
    },
    /// Show readers and writers of one state cell.
    State {
        manifest: PathBuf,
        cell: String,
    },
    /// Validate a manifest. Exits non-zero on failure.
    Validate { manifest: PathBuf },
    /// Emit the generated WIT world for one handler.
    Wit {
        manifest: PathBuf,
        handler: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, Box<dyn std::error::Error>> {
    match cli.command {
        Cmd::Show { manifest } => {
            let m = load_manifest(&manifest)?;
            print!("{}", program_map(&m));
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Handler { manifest, handler } => {
            let m = load_manifest(&manifest)?;
            match handler_card(&m, &handler) {
                Some(text) => {
                    print!("{text}");
                    Ok(ExitCode::SUCCESS)
                }
                None => {
                    eprintln!("no such handler: {handler}");
                    Ok(ExitCode::from(1))
                }
            }
        }
        Cmd::State { manifest, cell } => {
            let m = load_manifest(&manifest)?;
            match state_cell(&m, &cell) {
                Some(text) => {
                    print!("{text}");
                    Ok(ExitCode::SUCCESS)
                }
                None => {
                    eprintln!("no such state cell: {cell}");
                    Ok(ExitCode::from(1))
                }
            }
        }
        Cmd::Validate { manifest } => {
            let m = load_manifest(&manifest)?;
            let (ok, text) = validate_report(&m);
            print!("{text}");
            Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) })
        }
        Cmd::Wit { manifest, handler } => {
            let m = load_manifest(&manifest)?;
            match m.handlers.iter().find(|h| h.name == handler) {
                Some(h) => {
                    print!("{}", ir::wit::generate_world(h, &m));
                    Ok(ExitCode::SUCCESS)
                }
                None => {
                    eprintln!("no such handler: {handler}");
                    Ok(ExitCode::from(1))
                }
            }
        }
    }
}

fn load_manifest(path: &PathBuf) -> Result<Manifest, Box<dyn std::error::Error>> {
    Ok(load_manifest_file(path)?)
}
