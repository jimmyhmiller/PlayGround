use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use diffpack::{Revision, RevisionResult, run_revisions, scan_graph};

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
    "usage: diffpack build <entry.js> | diffpack demo [fixture-root]".into()
}
