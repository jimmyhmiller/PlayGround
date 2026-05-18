//! Generic runner for constrained-language programs.
//!
//! Takes a single program file (TOML) containing both the IR (schemas,
//! events, state, effects, handlers) and the host wiring (adapters,
//! generators). Loads the manifest, wires up handler bodies, adapters, and
//! generators, then runs the runtime until the event queue is empty AND
//! every declared generator has finished producing.
//!
//! No "modes" — the program's runtime shape is defined by what generators
//! it declares. A program with a `stdin_lines` generator is interactive
//! because the generator blocks on stdin. A program with no generators and
//! a pre-enqueued event is one-shot. Everything in between composes.

mod adapters;
mod config;
mod generators;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use ir::load_manifest_file;
use runtime::wasm::load_handler_body;
use runtime::Runtime;

#[derive(Parser)]
#[command(name = "cl-run")]
#[command(about = "Run a constrained-language program")]
struct Cli {
    /// Path to the program manifest (TOML).
    program: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli.program) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cl-run: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(program_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // The program file is parsed twice: once as `ir::Manifest` (which
    // silently ignores host-wiring sections), once as the runner-only extras.
    let program_dir = program_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest = load_manifest_file(program_path)?;
    let runner = config::load(program_path)?;

    let mut rt = Runtime::new(manifest.clone())?;

    let components_dir = runner
        .components_dir
        .as_ref()
        .map(|p| resolve_relative(program_dir, p))
        .unwrap_or_else(|| program_dir.to_path_buf());

    // ---- 1. Handler bodies ----
    for handler in &manifest.handlers {
        let component_path = resolve_component(&components_dir, &handler.body.uri)?;
        let mut body = load_handler_body(&manifest, &handler.name, &component_path)?;
        let key = handler.body.uri.clone();
        rt.bodies.register(key, move |ctx| body(ctx));
    }

    // ---- 2. Adapters (one per declared effect) ----
    for (effect_name, cfg) in &runner.adapters {
        if !manifest.effects.contains_key(effect_name) {
            return Err(format!(
                "program file configures adapter for unknown effect `{effect_name}`"
            )
            .into());
        }
        let adapter = adapters::build(&cfg.kind, &cfg.options)?;
        rt.adapters.register(effect_name, adapter);
    }
    for effect in manifest.effects.keys() {
        if !runner.adapters.contains_key(effect) {
            return Err(
                format!("manifest declares effect `{effect}` but program file does not configure an adapter for it")
                    .into(),
            );
        }
    }

    // ---- 3. Generators (one per declared generator) ----
    for (gen_name, decl) in &manifest.generators {
        let g = generators::build(&decl.kind, decl.payload.clone())?;
        rt.generators.register(gen_name, g);
    }

    // ---- 4. Run ----
    rt.start_generators()?;
    rt.run_until_idle()?;

    Ok(())
}

fn resolve_relative(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Try `<dir>/<uri>.component.wasm` first, then `<dir>/<uri>/<uri>.component.wasm`.
fn resolve_component(dir: &Path, uri: &str) -> Result<PathBuf, String> {
    let flat = dir.join(format!("{uri}.component.wasm"));
    if flat.exists() {
        return Ok(flat);
    }
    let nested = dir.join(uri).join(format!("{uri}.component.wasm"));
    if nested.exists() {
        return Ok(nested);
    }
    Err(format!(
        "no component file for handler uri `{uri}` (tried {} and {})",
        flat.display(),
        nested.display()
    ))
}
