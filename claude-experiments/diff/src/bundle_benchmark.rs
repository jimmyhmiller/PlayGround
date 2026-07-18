use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::bundler::{Bundler, EmitOptions};
use crate::frontend_profile;
use crate::visualizer::write_visualization;

#[derive(Debug)]
pub struct BundleScaleResult {
    pub worker_threads: usize,
    pub modules: usize,
    pub imports_per_module: usize,
    pub generated_edges: usize,
    pub initial_reachable: usize,
    pub final_reachable: usize,
    pub source_bytes: u64,
    pub bundle_bytes: u64,
    pub runtime_value: Option<u64>,
    pub generate_ms: f64,
    pub discover_transform_resolve_ms: f64,
    pub initial_reachability_ms: f64,
    pub initial_emit_ms: f64,
    pub edit_transform_resolve_ms: f64,
    pub edit_reachability_ms: f64,
    pub edit_emit_ms: f64,
    pub transformed_on_edit: usize,
    pub frontend_read_cpu_ms: f64,
    pub frontend_transform_cpu_ms: f64,
    pub frontend_lower_cpu_ms: f64,
    pub frontend_resolve_cpu_ms: f64,
}

pub fn run_bundle_scale_direct(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_impl(module_count, imports_per_module, false, false, false)
}

pub fn run_bundle_scale_direct_dependency_edit(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_impl(module_count, imports_per_module, true, false, false)
}

pub fn run_bundle_scale_direct_live(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_with_live_output(module_count, imports_per_module, false)
}

pub fn run_bundle_scale_direct_live_dependency_edit(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_with_live_output(module_count, imports_per_module, true)
}

pub fn run_bundle_scale_direct_live_minified(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_with_live_output_and_minify(module_count, imports_per_module, false)
}

pub fn run_bundle_scale_direct_live_minified_dependency_edit(
    module_count: usize,
    imports_per_module: usize,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_with_live_output_and_minify(module_count, imports_per_module, true)
}

pub fn write_live_scale_visualization(
    module_count: usize,
    imports_per_module: usize,
    output: &std::path::Path,
) -> Result<(usize, usize, usize), String> {
    if module_count == 0 || imports_per_module == 0 {
        return Err("module count and imports per module must be positive".into());
    }
    let workspace = TemporaryProject::new()?;
    (0..module_count)
        .into_par_iter()
        .map(|index| {
            fs::write(
                workspace.path.join(module_name(index)),
                live_module_source(index, module_count, imports_per_module, index),
            )
            .map_err(|error| format!("cannot generate visualization module {index}: {error}"))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let entry = workspace.path.join(module_name(0));
    let (mut bundler, update) = Bundler::discover_direct(&entry)?;
    if !update.diagnostics.is_empty() {
        return Err(format!(
            "visualization build produced {} diagnostics; first: {}",
            update.diagnostics.len(),
            update.diagnostics[0]
        ));
    }
    let mut reachability = bundler.direct_reachability();
    fs::write(
        &entry,
        live_module_source_after_dependency_removal(
            0,
            module_count,
            imports_per_module,
            module_count,
        ),
    )
    .map_err(|error| format!("cannot edit visualization entry: {error}"))?;
    let revision = bundler.rebuild_path(&entry)?;
    reachability.apply(&revision.delta);
    let reachable = reachability.reachable_modules();
    let graph = bundler.visualization_graph(&reachable);
    let edge_count = graph.edges.len();
    write_visualization(&graph, output)?;
    Ok((graph.nodes.len(), edge_count, reachable.len()))
}

fn run_bundle_scale_inner_with_live_output(
    module_count: usize,
    imports_per_module: usize,
    dependency_edit: bool,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_impl(
        module_count,
        imports_per_module,
        dependency_edit,
        true,
        false,
    )
}

fn run_bundle_scale_inner_with_live_output_and_minify(
    module_count: usize,
    imports_per_module: usize,
    dependency_edit: bool,
) -> Result<BundleScaleResult, String> {
    run_bundle_scale_inner_impl(
        module_count,
        imports_per_module,
        dependency_edit,
        true,
        true,
    )
}

fn run_bundle_scale_inner_impl(
    module_count: usize,
    imports_per_module: usize,
    dependency_edit: bool,
    live_output: bool,
    minify: bool,
) -> Result<BundleScaleResult, String> {
    if module_count == 0 || imports_per_module == 0 {
        return Err("module count and imports per module must be positive".into());
    }
    let workspace = TemporaryProject::new()?;
    let generation_started = Instant::now();
    let generated = (0..module_count)
        .into_par_iter()
        .map(|index| {
            let source = if live_output {
                live_module_source(index, module_count, imports_per_module, index)
            } else {
                module_source(index, module_count, imports_per_module, index)
            };
            let edges = if live_output {
                live_dependency_indices(index, module_count, imports_per_module).len()
            } else {
                dependency_indices(index, module_count, imports_per_module).len()
            };
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
    frontend_profile::reset();
    let discover_started = Instant::now();
    let (mut bundler, initial) = Bundler::discover_direct(&entry)?;
    let discover_transform_resolve_ms = elapsed_ms(discover_started);
    let frontend_profile = frontend_profile::snapshot();
    if !initial.diagnostics.is_empty() {
        return Err(format!(
            "initial build produced {} diagnostics; first: {}",
            initial.diagnostics.len(),
            initial.diagnostics[0]
        ));
    }

    let reachability_started = Instant::now();
    let mut reachability = bundler.direct_reachability();
    let mut reachable = reachability.reachable_modules();
    let initial_reachable = reachable.len();
    let initial_reachability_ms = elapsed_ms(reachability_started);

    let emit_started = Instant::now();
    bundler.emit_with_options(
        &reachable,
        &output,
        EmitOptions {
            source_map: false,
            minify,
        },
    )?;
    let initial_emit_ms = elapsed_ms(emit_started);

    let edited_index = if dependency_edit { 0 } else { module_count / 2 };
    let edited_path = workspace.path.join(module_name(edited_index));
    let edited_value = module_count + edited_index;
    let edited_source = if live_output && dependency_edit {
        live_module_source_after_dependency_removal(
            edited_index,
            module_count,
            imports_per_module,
            edited_value,
        )
    } else if live_output {
        live_module_source(edited_index, module_count, imports_per_module, edited_value)
    } else if dependency_edit {
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

    let edit_reachability_started = Instant::now();
    let edit_result = reachability.apply(&update.delta);
    for removed in edit_result.removed {
        reachable.remove(&removed);
    }
    reachable.extend(edit_result.added);
    let edit_reachability_ms = elapsed_ms(edit_reachability_started);
    let final_reachable = reachable.len();

    let edit_emit_started = Instant::now();
    bundler.emit_with_options(
        &reachable,
        &output,
        EmitOptions {
            source_map: false,
            minify,
        },
    )?;
    let edit_emit_ms = elapsed_ms(edit_emit_started);
    let bundle_bytes = fs::metadata(&output)
        .map_err(|error| format!("cannot inspect benchmark bundle: {error}"))?
        .len();
    let runtime_value = if live_output {
        let execution = std::process::Command::new("node")
            .arg(&output)
            .output()
            .map_err(|error| format!("cannot execute benchmark bundle: {error}"))?;
        if !execution.status.success() {
            return Err(format!(
                "benchmark bundle failed at runtime: {}",
                String::from_utf8_lossy(&execution.stderr).trim()
            ));
        }
        let stdout = String::from_utf8(execution.stdout)
            .map_err(|error| format!("benchmark output is not UTF-8: {error}"))?;
        Some(
            stdout
                .trim()
                .parse::<u64>()
                .map_err(|error| format!("benchmark output is not an integer: {error}"))?,
        )
    } else {
        None
    };

    Ok(BundleScaleResult {
        worker_threads: rayon::current_num_threads(),
        modules: module_count,
        imports_per_module,
        generated_edges,
        initial_reachable,
        final_reachable,
        source_bytes,
        bundle_bytes,
        runtime_value,
        generate_ms,
        discover_transform_resolve_ms,
        initial_reachability_ms,
        initial_emit_ms,
        edit_transform_resolve_ms,
        edit_reachability_ms,
        edit_emit_ms,
        transformed_on_edit,
        frontend_read_cpu_ms: frontend_profile.read_ms,
        frontend_transform_cpu_ms: frontend_profile.transform_ms,
        frontend_lower_cpu_ms: frontend_profile.lower_ms,
        frontend_resolve_cpu_ms: frontend_profile.resolve_ms,
    })
}

/// Deterministic memory accounting for a build plus a run of incremental edits.
/// Every field is a byte delta against a pre-build baseline, so it is
/// machine-independent and safe to assert on.
#[derive(Debug)]
pub struct MemoryScaleResult {
    pub modules: usize,
    pub reachable: usize,
    pub source_bytes: u64,
    /// Peak live bytes reached during the initial build, above baseline.
    pub build_peak_bytes: usize,
    /// Live bytes still held once the graph is built (the resident cost of the
    /// module graph), above baseline.
    pub retained_after_build_bytes: usize,
    pub bytes_per_module: f64,
    pub edits: usize,
    /// The most modules re-transformed by any single edit. Incrementality holds
    /// when a leaf edit re-transforms exactly one module.
    pub transformed_per_edit_max: usize,
    /// Growth in live bytes across all incremental edits. Bounded growth means
    /// old revisions are released rather than accumulated.
    pub retained_growth_over_edits_bytes: usize,
    /// Live bytes still held after the bundler is dropped, above baseline (a
    /// leak proxy; should be near zero).
    pub retained_after_drop_bytes: usize,
}

/// Builds a synthetic graph, edits one leaf module `edits` times, and reports
/// deterministic allocation deltas for the build, the edit run, and teardown.
pub fn run_bundle_scale_memory(
    module_count: usize,
    imports_per_module: usize,
    edits: usize,
) -> Result<MemoryScaleResult, String> {
    if module_count == 0 || imports_per_module == 0 {
        return Err("module count and imports per module must be positive".into());
    }
    let workspace = TemporaryProject::new()?;
    let source_bytes = (0..module_count)
        .into_par_iter()
        .map(|index| {
            let source = module_source(index, module_count, imports_per_module, index);
            let bytes = source.len() as u64;
            fs::write(workspace.path.join(module_name(index)), source)
                .map_err(|error| format!("cannot generate module {index}: {error}"))?;
            Ok(bytes)
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .sum();

    let entry = workspace.path.join(module_name(0));
    let output = workspace.path.join("dist/bundle.js");

    // Baseline after generation, before any graph exists.
    let baseline = crate::memory::snapshot().live_bytes;
    crate::memory::reset_peak();

    let (mut bundler, initial) = Bundler::discover_direct(&entry)?;
    if !initial.diagnostics.is_empty() {
        return Err(format!(
            "memory build produced {} diagnostics; first: {}",
            initial.diagnostics.len(),
            initial.diagnostics[0]
        ));
    }
    let mut reachability = bundler.direct_reachability();
    let mut reachable = reachability.reachable_modules();
    bundler.emit(&reachable, &output)?;
    let reachable_count = reachable.len();

    let after_build = crate::memory::snapshot();
    let build_peak_bytes = after_build.peak_bytes.saturating_sub(baseline);
    let retained_after_build_bytes = after_build.live_bytes.saturating_sub(baseline);
    let live_after_build = after_build.live_bytes;

    // Edit one leaf module repeatedly. A correct incremental graph re-transforms
    // only that module and releases the prior revision, so live memory must not
    // grow with the number of edits.
    let edited_index = module_count / 2;
    let edited_path = workspace.path.join(module_name(edited_index));
    let mut transformed_per_edit_max = 0;
    for edit in 0..edits {
        let value = module_count + edit + 1;
        fs::write(
            &edited_path,
            module_source(edited_index, module_count, imports_per_module, value),
        )
        .map_err(|error| format!("cannot edit memory-benchmark module: {error}"))?;
        let update = bundler.rebuild_path(&edited_path)?;
        transformed_per_edit_max = transformed_per_edit_max.max(update.transformed_modules);
        let result = reachability.apply(&update.delta);
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        bundler.emit(&reachable, &output)?;
    }

    let live_after_edits = crate::memory::snapshot().live_bytes;
    let retained_growth_over_edits_bytes = live_after_edits.saturating_sub(live_after_build);

    drop(reachable);
    drop(reachability);
    drop(bundler);
    let retained_after_drop_bytes = crate::memory::snapshot().live_bytes.saturating_sub(baseline);

    Ok(MemoryScaleResult {
        modules: module_count,
        reachable: reachable_count,
        source_bytes,
        build_peak_bytes,
        retained_after_build_bytes,
        bytes_per_module: retained_after_build_bytes as f64 / module_count as f64,
        edits,
        transformed_per_edit_max,
        retained_growth_over_edits_bytes,
        retained_after_drop_bytes,
    })
}

fn live_module_source(
    index: usize,
    module_count: usize,
    imports_per_module: usize,
    value: usize,
) -> String {
    let dependencies = live_dependency_indices(index, module_count, imports_per_module);
    let mut source = String::new();
    for dependency in &dependencies {
        source.push_str(&format!(
            "import {{ value_{dependency} }} from \"./{}\";\n",
            module_name(*dependency)
        ));
    }
    let terms = dependencies
        .iter()
        .map(|dependency| format!("value_{dependency}"))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        source.push_str(&format!("export const value_{index}: number = {value};\n"));
    } else {
        source.push_str(&format!(
            "export const value_{index}: number = {value} + {};\n",
            terms.join(" + ")
        ));
    }
    if index == 0 {
        source.push_str("console.log(value_0);\n");
    }
    source
}

fn live_module_source_after_dependency_removal(
    index: usize,
    module_count: usize,
    imports_per_module: usize,
    value: usize,
) -> String {
    let mut dependencies = live_dependency_indices(index, module_count, imports_per_module);
    if !dependencies.is_empty() {
        dependencies.remove(0);
    }
    let mut source = String::new();
    for dependency in &dependencies {
        source.push_str(&format!(
            "import {{ value_{dependency} }} from \"./{}\";\n",
            module_name(*dependency)
        ));
    }
    let terms = dependencies
        .iter()
        .map(|dependency| format!("value_{dependency}"))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        source.push_str(&format!("export const value_{index}: number = {value};\n"));
    } else {
        source.push_str(&format!(
            "export const value_{index}: number = {value} + {};\n",
            terms.join(" + ")
        ));
    }
    if index == 0 {
        source.push_str("console.log(value_0);\n");
    }
    source
}

fn live_dependency_indices(
    index: usize,
    module_count: usize,
    imports_per_module: usize,
) -> Vec<usize> {
    (1..=imports_per_module)
        .map(|offset| index * imports_per_module + offset)
        .take_while(|child| *child < module_count)
        .collect()
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

#[cfg(test)]
mod thesis_guards {
    //! Parallelism-safe incrementality guard. The memory-delta guards live in
    //! `tests/thesis_memory.rs` because they read process-wide allocation
    //! counters and must run isolated from other tests' allocations.

    use super::run_bundle_scale_direct;

    #[test]
    fn a_leaf_edit_retransforms_exactly_one_module() {
        let result = run_bundle_scale_direct(600, 4).unwrap();
        assert_eq!(
            result.transformed_on_edit, 1,
            "editing one leaf module must re-transform exactly that module, not {} of {}",
            result.transformed_on_edit, result.modules
        );
    }
}
