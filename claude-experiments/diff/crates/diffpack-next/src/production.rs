//! Next-owned production assembly and prerender entry points.

use std::path::Path;

use diffpack_default_loader::driver::EmitOptions;

const NEXT_RENDER_CORE_MJS: &str = include_str!("../../../scripts/rsc/next-render-core.mjs");
const NEXT_PRERENDER_MJS: &str = include_str!("../../../scripts/rsc/next-prerender.mjs");
const SSR_MODULE_MAP_MJS: &str = include_str!("../../../scripts/rsc/ssr-module-map.mjs");
const NEXT_SERVER_MJS: &str = include_str!("../../../scripts/rsc/next-server.mjs");

pub fn prerender_pages(output_root: &Path) -> Result<(), String> {
    let driver = output_root.join("pages-prerender.mjs");
    std::fs::write(&driver, crate::next_pages::PRERENDER_DRIVER)
        .map_err(|error| format!("cannot write {}: {error}", driver.display()))?;
    let status = std::process::Command::new("node")
        .arg(&driver)
        .arg(output_root)
        .status()
        .map_err(|error| format!("cannot spawn node for the pages SSG prerenderer: {error}"))?;
    if !status.success() {
        return Err(format!(
            "the pages SSG prerenderer (node {}) failed with {status}",
            driver.display()
        ));
    }
    Ok(())
}

pub fn write_pages_server(output_root: &Path) -> Result<(), String> {
    let path = output_root.join("pages-server.mjs");
    std::fs::write(&path, crate::next_pages::ORCHESTRATOR)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

pub fn build_pages_environment(
    project_root: &Path,
    environment: &str,
    minify: bool,
    source_map_override: Option<bool>,
) -> Result<(), String> {
    let mut config = crate::next_pages::configure(project_root, environment, false)?
        .ok_or_else(|| "Next Pages configuration returned no profile".to_string())?;
    if let Some(source_maps) = source_map_override {
        config.build.source_maps = source_maps;
    }
    let source_maps = config.build.source_maps;
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| format!("no {environment} entry found for the Pages app"))?;
    let output_root = project_root.join(".diffpack-output");
    let (bundler, update) = crate::compiler::discover(&entry, &config.build)?;
    let warnings = diffpack_core::partition_diagnostics(
        &update.diagnostics,
        &format!("Pages {} build", config.environment),
    )?;
    for warning in warnings {
        println!("warning: {warning}");
    }
    let reachable = bundler.reachable_modules_direct();
    let options = EmitOptions {
        minify,
        source_map: source_maps,
        ..EmitOptions::default()
    };
    let summary = if config.environment == "client" {
        let summary = bundler.emit_public(&reachable, &output_root, options)?;
        let static_files =
            diffpack_web::config::copy_static_public(project_root, &summary.output_dir)?;
        println!("copied {static_files} static file(s)");
        summary
    } else {
        bundler.emit_server(&reachable, &output_root, options)?
    };
    println!(
        "emitted {}: {} JavaScript, {} CSS, {} asset file(s)",
        summary.output_dir.display(),
        summary.javascript_files,
        summary.css_files,
        summary.asset_files
    );
    Ok(())
}

pub fn write_ssr_module_map(output_root: &Path) -> Result<(), String> {
    let path = output_root.join(crate::rsc::SSR_MODULE_MAP_FILE);
    std::fs::write(&path, SSR_MODULE_MAP_MJS)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// Completes the Next-owned deployable server layout after all environment builds.
pub fn assemble_server(output_root: &Path) -> Result<(), String> {
    let rsc_assets = output_root.join("rsc-render/assets");
    if rsc_assets.is_dir() {
        copy_dir_recursive(&rsc_assets, &output_root.join("public/assets"))?;
    }
    let server = output_root.join("next-server.mjs");
    std::fs::write(&server, NEXT_SERVER_MJS)
        .map_err(|error| format!("cannot write {}: {error}", server.display()))?;
    write_ssr_module_map(output_root)
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::create_dir_all(destination)
        .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
    for entry in std::fs::read_dir(source)
        .map_err(|error| format!("cannot read {}: {error}", source.display()))?
    {
        let entry =
            entry.map_err(|error| format!("cannot read entry in {}: {error}", source.display()))?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", from.display()))?;
        if kind.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).map_err(|error| {
                format!(
                    "cannot copy {} -> {}: {error}",
                    from.display(),
                    to.display()
                )
            })?;
        }
    }
    Ok(())
}

pub fn build_static(project_root: &Path, static_export: bool) -> Result<(), String> {
    let output_root = project_root.join(".diffpack-output");
    for (label, rel) in [
        ("react-server render bundle", "rsc-render/server.mjs"),
        ("SSR bundle", "server/server.mjs"),
        (
            "client-references manifest",
            crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE,
        ),
        (
            "react-server-references manifest",
            crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE,
        ),
        (
            "ssr-references manifest",
            crate::rsc::SERVER_REFERENCES_MANIFEST_FILE,
        ),
    ] {
        let path = output_root.join(rel);
        if !path.exists() {
            return Err(format!(
                "{label} not found at {} — run the client -> react-server (cp -> rsc-render) -> ssr builds first",
                path.display()
            ));
        }
    }
    prerender_app(project_root, &output_root, static_export)?;
    println!(
        "next SSG: prerendered static routes -> {}",
        output_root.join("static").display()
    );
    Ok(())
}

pub fn prerender_app(
    project_root: &Path,
    output_root: &Path,
    static_export: bool,
) -> Result<(), String> {
    let plan_stage = diffpack_core::build_profile::stage("prerender/classify-routes");
    let route_count = crate::next_adapter::write_prerender_plan(project_root, output_root)?;
    drop(plan_stage);
    println!(
        "next SSG/ISR: classified {route_count} route(s) -> {}",
        output_root.join("static/prerender-plan.json").display()
    );
    let core_path = output_root.join("next-render-core.mjs");
    let prerender_path = output_root.join("next-prerender.mjs");
    std::fs::write(&core_path, NEXT_RENDER_CORE_MJS)
        .map_err(|error| format!("cannot write {}: {error}", core_path.display()))?;
    std::fs::write(&prerender_path, NEXT_PRERENDER_MJS)
        .map_err(|error| format!("cannot write {}: {error}", prerender_path.display()))?;
    write_ssr_module_map(output_root)?;

    let mut command = std::process::Command::new("node");
    command.arg(&prerender_path).arg(output_root);
    command.envs(crate::next_adapter::config_env_from_manifest(project_root));
    if static_export {
        command.arg("--static-export");
    }
    let render_stage = diffpack_core::build_profile::stage("prerender/render-routes");
    let status = command
        .status()
        .map_err(|error| format!("cannot spawn node for the SSG prerenderer: {error}"))?;
    drop(render_stage);
    if !status.success() {
        return Err(format!(
            "the SSG prerenderer (node {}) failed with {status}",
            prerender_path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_entry_point_names_the_first_missing_framework_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let error = build_static(directory.path(), false).unwrap_err();
        assert!(error.contains("react-server render bundle"), "{error}");
        assert!(error.contains("rsc-render/server.mjs"), "{error}");
    }
}
