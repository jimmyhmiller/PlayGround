//! Final TanStack/Vite profile assembly performed before graph discovery.

use std::collections::BTreeSet;
use std::path::Path;

use diffpack_default_loader::driver::{Bundler, EmitOptions};
use diffpack_default_loader::driver_config::EnvironmentConfig;

pub fn prepare_build(
    project_root: &Path,
    output_root: &Path,
    config: &mut EnvironmentConfig,
) -> Result<(), String> {
    if let Some(route_count) = crate::route_tree::generate_for_project(project_root)? {
        println!(
            "generated src/{} natively ({route_count} route(s))",
            crate::route_tree::ROUTE_TREE_FILE
        );
    }
    if config.environment == "client" {
        return Ok(());
    }

    let client_manifest_path = output_root.join(crate::manifest::CLIENT_MANIFEST_FILE);
    let client_manifest = crate::manifest::ClientRouteManifest::read(&client_manifest_path)?;
    config.build.virtual_modules.push((
        crate::manifest::START_MANIFEST_SPECIFIER.to_string(),
        client_manifest.to_start_manifest_source(),
    ));
    config.build.virtual_modules.push((
        crate::manifest::INJECTED_HEAD_SCRIPTS_SPECIFIER.to_string(),
        crate::manifest::injected_head_scripts_module_source(),
    ));
    println!(
        "loaded client route manifest ({} routes) from {}",
        client_manifest.routes.len(),
        client_manifest_path.display()
    );

    let server_fns = crate::server_fn::scan_project_server_fns(project_root)?;
    config.build.virtual_modules.push((
        crate::server_fn::RESOLVER_SPECIFIER.to_string(),
        crate::server_fn::generate_resolver_module(&server_fns),
    ));
    println!(
        "registered {} server function(s) in the native server-fn resolver",
        server_fns.len()
    );
    Ok(())
}

pub fn emit_build(
    project_root: &Path,
    output_root: &Path,
    config: &EnvironmentConfig,
    bundler: &Bundler,
    reachable: &BTreeSet<String>,
    emit_options: EmitOptions,
    server_dir_name: &str,
) -> Result<(), String> {
    if config.environment == "client" {
        let manifest = crate::manifest::from_bundle_graph(
            &bundler.integration_manifest_graph(reachable, "client.js")?,
            "/",
        )?;
        let manifest_path = output_root.join(crate::manifest::CLIENT_MANIFEST_FILE);
        std::fs::create_dir_all(output_root)
            .map_err(|error| format!("cannot create {}: {error}", output_root.display()))?;
        let staged = manifest_path.with_extension(format!("staged-{}", std::process::id()));
        manifest.write(&staged)?;
        std::fs::rename(&staged, &manifest_path)
            .map_err(|error| format!("cannot publish {}: {error}", manifest_path.display()))?;
        let summary = bundler.emit_public(reachable, output_root, emit_options)?;
        let static_files =
            diffpack_web::config::copy_static_public(project_root, &summary.output_dir)?;
        println!(
            "emitted {}: {} public .js, {} .css, {} asset(s), {} static file(s)",
            summary.output_dir.display(),
            summary.javascript_files,
            summary.css_files,
            summary.asset_files,
            static_files
        );
        println!(
            "wrote {} ({} routes mapped to client chunks)",
            manifest_path.display(),
            manifest.routes.len()
        );
    } else {
        let summary = bundler.emit_server_into(
            reachable,
            &output_root.join(server_dir_name),
            emit_options,
        )?;
        println!(
            "emitted {}: {} server .mjs, {} .css, {} asset(s)",
            summary.output_dir.display(),
            summary.javascript_files,
            summary.css_files,
            summary.asset_files
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_client_profile_entry_point_needs_no_vite_or_manifest_file() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = EnvironmentConfig {
            environment: "client".into(),
            build: Default::default(),
            entry: None,
        };
        prepare_build(directory.path(), directory.path(), &mut config).unwrap();
        assert!(config.build.virtual_modules.is_empty());
    }
}
