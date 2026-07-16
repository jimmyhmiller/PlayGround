use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::DeltaSession;
use crate::bundler::Bundler;

#[derive(Debug)]
pub struct BundleScaleResult {
    pub worker_threads: usize,
    pub dataflow_threads: usize,
    pub modules: usize,
    pub imports_per_module: usize,
    pub generated_edges: usize,
    pub initial_reachable: usize,
    pub final_reachable: usize,
    pub source_bytes: u64,
    pub bundle_bytes: u64,
    pub generate_ms: f64,
    pub discover_transform_resolve_ms: f64,
    pub initial_dataflow_ms: f64,
    pub initial_emit_ms: f64,
    pub edit_transform_resolve_ms: f64,
    pub edit_dataflow_ms: f64,
    pub edit_emit_ms: f64,
    pub transformed_on_edit: usize,
}

pub fn run_bundle_scale(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner(module_count, imports_per_module, true, false)
}

pub fn run_bundle_scale_direct(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner(module_count, imports_per_module, false, false)
}

pub fn run_bundle_scale_dependency_edit(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner(module_count, imports_per_module, true, true)
}

pub fn run_bundle_scale_direct_dependency_edit(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner(module_count, imports_per_module, false, true)
}

fn run_bundle_scale_inner(
    module_count: usize,
    imports_per_module: usize,
    use_dataflow: bool,
    dependency_edit: bool,
) -> Result<BundleScaleResult, String> {
    if module_count == 0 || imports_per_module == 0 {
        return Err("module count and imports per module must be positive".into());
    }
    let workspace = TemporaryProject::new()?;
    let generation_started = Instant::now();
    let generated = (0..module_count)
        .into_par_iter()
        .map(|index| {
            let source = module_source(index, module_count, imports_per_module, index);
            let edges = dependency_indices(index, module_count, imports_per_module).len();
            let bytes = source.len() as u64;
            fs::write(workspace.path.join(module_name(index)), source)
                .map_err(|error| format!("cannot generate module {index}: {error}"))?;
            Ok((edges, bytes))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let generated_edges = generated.iter().map(|(edges, _)| edges).sum();
    let source_bytes = generated.iter().map(|(_, bytes)| bytes).sum();
    let generate_ms = elapsed_ms(generation_started);

    let entry = workspace.path.join(module_name(0));
    let output = workspace.path.join("dist/bundle.js");
    let discover_started = Instant::now();
    let (mut bundler, initial) = Bundler::discover(&entry)?;
    let discover_transform_resolve_ms = elapsed_ms(discover_started);
    if !initial.diagnostics.is_empty() {
        return Err(format!(
            "initial build produced {} diagnostics; first: {}",
            initial.diagnostics.len(),
            initial.diagnostics[0]
        ));
    }

    let session = use_dataflow.then(DeltaSession::new);
    let dataflow_threads = session.as_ref().map_or(0, DeltaSession::worker_count);
    let dataflow_started = Instant::now();
    let mut direct_session = None;
    let mut reachable = if let Some(session) = &session {
        session.apply(initial.delta)?.added
    } else {
        let direct = bundler.direct_reachability();
        let reachable = direct.reachable_modules();
        direct_session = Some(direct);
        reachable
    };
    let initial_reachable = reachable.len();
    let initial_dataflow_ms = elapsed_ms(dataflow_started);

    let emit_started = Instant::now();
    bundler.emit(&reachable, &output)?;
    let initial_emit_ms = elapsed_ms(emit_started);

    let edited_index = if dependency_edit { 0 } else { module_count / 2 };
    let edited_path = workspace.path.join(module_name(edited_index));
    let edited_source = if dependency_edit {
        module_source_after_dependency_removal(
            edited_index,
            module_count,
            imports_per_module,
            usize::MAX,
        )
    } else {
        module_source(edited_index, module_count, imports_per_module, usize::MAX)
    };
    fs::write(&edited_path, edited_source)
        .map_err(|error| format!("cannot edit benchmark module: {error}"))?;
    let edit_started = Instant::now();
    let update = bundler.rebuild_path(&edited_path)?;
    let edit_transform_resolve_ms = elapsed_ms(edit_started);
    let transformed_on_edit = update.transformed_modules;

    let edit_dataflow_started = Instant::now();
    if let Some(session) = &session {
        let edit_result = session.apply(update.delta)?;
        for removed in edit_result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(edit_result.added);
    } else {
        let edit_result = direct_session
            .as_mut()
            .expect("direct reachability session must exist")
            .apply(&update.delta);
        for removed in edit_result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(edit_result.added);
    }
    let edit_dataflow_ms = elapsed_ms(edit_dataflow_started);
    let final_reachable = reachable.len();

    let edit_emit_started = Instant::now();
    bundler.emit(&reachable, &output)?;
    let edit_emit_ms = elapsed_ms(edit_emit_started);
    let bundle_bytes = fs::metadata(&output)
        .map_err(|error| format!("cannot inspect benchmark bundle: {error}"))?
        .len();

    Ok(BundleScaleResult {
        worker_threads: rayon::current_num_threads(),
        dataflow_threads,
        modules: module_count,
        imports_per_module,
        generated_edges,
        initial_reachable,
        final_reachable,
        source_bytes,
        bundle_bytes,
        generate_ms,
        discover_transform_resolve_ms,
        initial_dataflow_ms,
        initial_emit_ms,
        edit_transform_resolve_ms,
        edit_dataflow_ms,
        edit_emit_ms,
        transformed_on_edit,
    })
}

fn module_source(
    index: usize,
    module_count: usize,
    imports_per_module: usize,
    value: usize,
) -> String {
    let mut source = String::new();
    for dependency in dependency_indices(index, module_count, imports_per_module) {
        source.push_str(&format!("import \"./{}\";\n", module_name(dependency)));
    }
    source.push_str(&format!("export const value_{index}: number = {value};\n"));
    source
}

fn module_source_after_dependency_removal(
    index: usize,
    module_count: usize,
    imports_per_module: usize,
    value: usize,
) -> String {
    let mut dependencies = dependency_indices(index, module_count, imports_per_module);
    if !dependencies.is_empty() {
        dependencies.remove(0);
    }
    let mut source = String::new();
    for dependency in dependencies {
        source.push_str(&format!("import \"./{}\";\n", module_name(dependency)));
    }
    source.push_str(&format!("export const value_{index}: number = {value};\n"));
    source
}

fn dependency_indices(index: usize, module_count: usize, imports_per_module: usize) -> Vec<usize> {
    const FANOUT: usize = 8;
    let mut dependencies = Vec::new();
    for offset in 1..=FANOUT {
        let child = index * FANOUT + offset;
        if child < module_count {
            dependencies.push(child);
        }
    }
    if index > 0 {
        let mut salt = 1_usize;
        while dependencies.len() < imports_per_module && salt <= index {
            let target = index - salt;
            if !dependencies.contains(&target) {
                dependencies.push(target);
            }
            salt += 1;
        }
    }
    dependencies.truncate(imports_per_module);
    dependencies
}

fn module_name(index: usize) -> String {
    format!("module-{index:08}.ts")
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

struct TemporaryProject {
    path: PathBuf,
}

impl TemporaryProject {
    fn new() -> Result<Self, String> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock error: {error}"))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "diffpack-bundle-scale-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path)
            .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TemporaryProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
