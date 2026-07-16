use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;

use diffpack::benchmark::synthetic_revisions;
use diffpack::bundle_benchmark::{
    run_bundle_scale, run_bundle_scale_dependency_edit, run_bundle_scale_direct,
    run_bundle_scale_direct_dependency_edit,
};
use diffpack::bundler::Bundler;
use diffpack::{
    DeltaSession, Revision, RevisionResult, run_delta_revisions, run_revisions, scan_graph,
};
use notify::{RecursiveMode, Watcher};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    match arguments.next().as_deref().and_then(|value| value.to_str()) {
        Some("build") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let snapshot = scan_graph(Path::new(&entry))?;
            print_results(run_revisions(vec![Revision {
                label: "initial".into(),
                snapshot,
            }]));
            Ok(())
        }
        Some("demo") => {
            let root = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("fixtures/incremental"));
            let revisions = ["01-initial", "02-add-leaf", "03-remove-leaf"]
                .into_iter()
                .map(|label| {
                    Ok(Revision {
                        label: label.into(),
                        snapshot: scan_graph(&root.join(label).join("entry.js"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            print_results(run_revisions(revisions));
            Ok(())
        }
        Some("scale") => {
            let modules = parse_usize(arguments.next(), 100_000, "module count")?;
            let fanout = parse_usize(arguments.next(), 8, "fanout")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            if modules == 0 || fanout == 0 || imports == 0 {
                return Err("module count, fanout, and imports must be positive".into());
            }
            println!("modules={modules} fanout={fanout} imports_per_module={imports}");
            println!(
                "revision,module_facts,edge_facts,reachable,input_update_ms,dataflow_ms,output_ms,added,changed,removed"
            );
            for result in run_delta_revisions(synthetic_revisions(modules, fanout, imports)) {
                println!(
                    "{},{},{},{},{:.3},{:.3},{:.3},{},{},{}",
                    result.label,
                    result.module_facts,
                    result.edge_facts,
                    result.reachable_facts,
                    result.input_update_micros as f64 / 1_000.0,
                    result.dataflow_micros as f64 / 1_000.0,
                    result.output_micros as f64 / 1_000.0,
                    result.added_facts,
                    result.changed_facts,
                    result.removed_facts,
                );
            }
            Ok(())
        }
        Some("bundle-scale") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(run_bundle_scale(modules, imports)?, "differential");
            Ok(())
        }
        Some("bundle-scale-direct") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(run_bundle_scale_direct(modules, imports)?, "direct");
            Ok(())
        }
        Some("bundle-scale-deps") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_dependency_edit(modules, imports)?,
                "differential-dependency-edit",
            );
            Ok(())
        }
        Some("bundle-scale-direct-deps") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_dependency_edit(modules, imports)?,
                "direct-dependency-edit",
            );
            Ok(())
        }
        Some("bundle") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let output = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dist/bundle.js"));
            let (bundler, update) = Bundler::discover(Path::new(&entry))?;
            let result = run_delta_revisions(vec![update.delta])
                .pop()
                .ok_or_else(|| "dataflow returned no build result".to_string())?;
            let reachable = bundler.all_modules();
            if result.reachable_facts != reachable.len() {
                return Err(format!(
                    "graph disagreement: dataflow reached {} modules, discovery loaded {}",
                    result.reachable_facts,
                    reachable.len()
                ));
            }
            bundler.emit(&reachable, &output)?;
            println!(
                "bundled {} modules to {} (transformed {})",
                reachable.len(),
                output.display(),
                update.transformed_modules
            );
            for diagnostic in update.diagnostics {
                eprintln!("diagnostic: {diagnostic}");
            }
            Ok(())
        }
        Some("watch") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let output = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dist/bundle.js"));
            watch_bundle(Path::new(&entry), &output)
        }
        _ => Err(usage()),
    }
}

fn print_results(results: Vec<RevisionResult>) {
    for result in results {
        println!("revision {}", result.label);
        println!(
            "  facts: modules={} edges={} reachable={}",
            result.module_facts, result.edge_facts, result.reachable_facts
        );
        print_delta('+', &result.added);
        print_delta('~', &result.changed);
        print_delta('-', &result.removed);
        println!("  manifest:");
        for module in result.reachable {
            println!("    {module}");
        }
        for diagnostic in result.diagnostics {
            println!("  diagnostic: {diagnostic}");
        }
    }
}

fn print_delta(marker: char, modules: &std::collections::BTreeSet<String>) {
    if !modules.is_empty() {
        println!(
            "  {marker} {}",
            modules.iter().cloned().collect::<Vec<_>>().join(", ")
        );
    }
}

fn usage() -> String {
    "usage: diffpack build <entry.js> | diffpack bundle <entry> [output] | diffpack watch <entry> [output] | diffpack demo [fixture-root] | diffpack scale [modules] [fanout] [imports-per-module] | diffpack bundle-scale [modules] [imports-per-module] | diffpack bundle-scale-direct [modules] [imports-per-module] | diffpack bundle-scale-deps [modules] [imports-per-module] | diffpack bundle-scale-direct-deps [modules] [imports-per-module]".into()
}

fn print_bundle_scale(result: diffpack::bundle_benchmark::BundleScaleResult, mode: &str) {
    println!(
        "mode,frontend_threads,dataflow_threads,modules,edges,initial_reachable,final_reachable,source_mb,bundle_mb,generate_ms,discover_transform_resolve_ms,initial_reachability_ms,initial_emit_ms,edit_transform_resolve_ms,edit_reachability_ms,edit_emit_ms,transformed_on_edit"
    );
    println!(
        "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        mode,
        result.worker_threads,
        result.dataflow_threads,
        result.modules,
        result.generated_edges,
        result.initial_reachable,
        result.final_reachable,
        result.source_bytes as f64 / 1_000_000.0,
        result.bundle_bytes as f64 / 1_000_000.0,
        result.generate_ms,
        result.discover_transform_resolve_ms,
        result.initial_dataflow_ms,
        result.initial_emit_ms,
        result.edit_transform_resolve_ms,
        result.edit_dataflow_ms,
        result.edit_emit_ms,
        result.transformed_on_edit,
    );
}

fn watch_bundle(entry: &Path, output: &Path) -> Result<(), String> {
    let (mut bundler, initial) = Bundler::discover(entry)?;
    let session = DeltaSession::new();
    let initial_result = session.apply(initial.delta)?;
    let mut reachable = initial_result.added;
    bundler.emit(&reachable, output)?;
    println!(
        "watching {} ({} modules); wrote {}",
        bundler.watch_root().display(),
        reachable.len(),
        output.display()
    );
    for diagnostic in initial.diagnostics {
        eprintln!("diagnostic: {diagnostic}");
    }

    let (events, receiver) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = events.send(event);
    })
    .map_err(|error| format!("cannot create filesystem watcher: {error}"))?;
    watcher
        .watch(&bundler.watch_root(), RecursiveMode::Recursive)
        .map_err(|error| format!("cannot start filesystem watcher: {error}"))?;

    loop {
        let event = receiver
            .recv()
            .map_err(|_| "filesystem watcher stopped".to_string())?
            .map_err(|error| format!("filesystem watch error: {error}"))?;
        let paths = event.paths.into_iter().collect::<BTreeSet<_>>();
        for path in paths {
            if !is_module_path(&path) {
                continue;
            }
            let update = bundler.rebuild_path(&path)?;
            if update.delta.entry_updates.is_empty()
                && update.delta.edge_updates.is_empty()
                && update.delta.module_updates.is_empty()
            {
                continue;
            }
            let result = session.apply(update.delta)?;
            for module in result.removed {
                reachable.remove(&module);
            }
            reachable.extend(result.added);
            bundler.emit(&reachable, output)?;
            println!(
                "rebuilt {}: reachable={} transformed={} dataflow={:.3}ms",
                path.display(),
                reachable.len(),
                update.transformed_modules,
                result.dataflow_micros as f64 / 1_000.0
            );
            for diagnostic in update.diagnostics {
                eprintln!("diagnostic: {diagnostic}");
            }
        }
    }
}

fn is_module_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "json")
    )
}

fn parse_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    description: &str,
) -> Result<usize, String> {
    value.map_or(Ok(default), |value| {
        value
            .to_string_lossy()
            .parse()
            .map_err(|error| format!("invalid {description}: {error}"))
    })
}
