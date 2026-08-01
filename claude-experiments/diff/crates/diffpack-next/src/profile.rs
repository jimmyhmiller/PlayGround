//! Final Next profile assembly performed before graph discovery.

use std::collections::BTreeSet;
use std::path::Path;

use diffpack_default_loader::driver::{Bundler, EmitOptions};
use diffpack_default_loader::driver_config::EnvironmentConfig;

pub fn prepare_build(
    project_root: &Path,
    output_root: &Path,
    config: &mut EnvironmentConfig,
) -> Result<(), String> {
    config.build.virtual_modules.push((
        crate::rsc::CALL_SERVER_SPECIFIER.to_string(),
        crate::rsc::call_server_module_source().to_string(),
    ));
    if config.environment == "client" {
        return Ok(());
    }

    let server_actions = crate::rsc::scan_project_server_actions(project_root)?;
    config.build.virtual_modules.push((
        crate::rsc::ACTION_RESOLVER_SPECIFIER.to_string(),
        crate::rsc::generate_action_resolver_module(&server_actions),
    ));
    config.build.virtual_modules.push((
        crate::rsc::ACTION_HANDLER_SPECIFIER.to_string(),
        crate::rsc::action_handler_module_source().to_string(),
    ));
    eprintln!(
        "registered {} server action(s) in the native rsc action resolver",
        server_actions.len()
    );

    let manifest = crate::rsc::ClientReferencesManifest::read(
        &output_root.join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE),
    )?;
    config.build.virtual_modules.push((
        crate::rsc::SSR_CONSUMER_MANIFEST_SPECIFIER.to_string(),
        manifest.to_ssr_consumer_manifest_module(None),
    ));
    eprintln!(
        "registered the rsc ssr consumer manifest ({} client reference(s))",
        manifest.entries.len()
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
        let references = crate::rsc::client_references_from_bundle_graph(
            &bundler.integration_manifest_graph(reachable, "client.js")?,
        );
        let references_path = output_root.join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
        std::fs::create_dir_all(output_root)
            .map_err(|error| format!("cannot create {}: {error}", output_root.display()))?;
        let staged = references_path.with_extension(format!("staged-{}", std::process::id()));
        references.write(&staged)?;
        std::fs::rename(&staged, &references_path)
            .map_err(|error| format!("cannot publish {}: {error}", references_path.display()))?;

        let summary = bundler.emit_public(reachable, output_root, emit_options)?;
        let static_files =
            diffpack_web::config::copy_static_public(project_root, &summary.output_dir)?;
        if let crate::next_adapter::ImageOptimization::Disabled(reason) =
            crate::next_adapter::ImageOptimization::for_project(project_root)
        {
            println!("next/image: {reason}, so no build-time variants are generated");
        }
        let public_images = crate::next_adapter::scan_public_images(project_root)?;
        if !public_images.is_empty() {
            let variants = crate::next_adapter::emit_image_variants(
                project_root,
                &summary.output_dir,
                &public_images,
            )?;
            if variants > 0 {
                println!("emitted {variants} next/image variant file(s)");
            }
        }
        let metadata_images =
            crate::next_adapter::emit_metadata_images(project_root, &summary.output_dir)?;
        let fonts = crate::next_font::emit_font_assets(project_root, &summary.output_dir)?;
        println!(
            "wrote {} ({} client reference(s)); emitted {} public .js, {} .css, {} asset(s), {} static file(s), {} metadata image(s), {} font(s)",
            references_path.display(),
            references.entries.len(),
            summary.javascript_files,
            summary.css_files,
            summary.asset_files,
            static_files,
            metadata_images,
            fonts
        );
        return Ok(());
    }

    let summary =
        bundler.emit_server_into(reachable, &output_root.join(server_dir_name), emit_options)?;
    if config.environment == "react-server" {
        let css = output_root
            .join(server_dir_name)
            .join(crate::next_adapter::RSC_EMITTED_CSS_FILE);
        if css.is_file() {
            let destination = output_root
                .join("public")
                .join(crate::next_adapter::RSC_CSS_URL.trim_start_matches('/'));
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            }
            std::fs::copy(&css, &destination).map_err(|error| {
                format!(
                    "cannot preserve react-server CSS to {}: {error}",
                    destination.display()
                )
            })?;
        }
    }
    let references = crate::rsc::client_references_from_bundle_graph(
        &bundler.integration_manifest_graph(reachable, "server.mjs")?,
    );
    let references_path = output_root.join(if config.environment == "react-server" {
        crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE
    } else {
        crate::rsc::SERVER_REFERENCES_MANIFEST_FILE
    });
    references.write(&references_path)?;
    println!(
        "wrote {} ({} client reference(s)); emitted {} server .mjs, {} .css, {} asset(s)",
        references_path.display(),
        references.entries.len(),
        summary.javascript_files,
        summary.css_files,
        summary.asset_files
    );
    Ok(())
}

/// Publish only a server graph's client-reference identity table.
///
/// Native Next owns server route emission itself, but its App Page runtime still
/// needs the SSR graph's module ids to decode Flight references. Producing that
/// graph fact must not force Diffpack's separate standalone server bundle to be
/// rendered as dead output.
pub fn emit_server_references(
    output_root: &Path,
    config: &EnvironmentConfig,
    bundler: &Bundler,
    reachable: &BTreeSet<String>,
) -> Result<(), String> {
    if config.environment == "client" {
        return Err("client references-only emission is not supported".to_string());
    }
    let references = crate::rsc::client_references_from_bundle_graph(
        &bundler.integration_manifest_graph(reachable, "server.mjs")?,
    );
    let path = output_root.join(if config.environment == "react-server" {
        crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE
    } else {
        crate::rsc::SERVER_REFERENCES_MANIFEST_FILE
    });
    references.write(&path)?;
    println!(
        "wrote {} ({} client reference(s)); skipped standalone server emission",
        path.display(),
        references.entries.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_client_profile_entry_point_registers_the_action_transport() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = EnvironmentConfig {
            environment: "client".into(),
            build: Default::default(),
            entry: None,
        };
        prepare_build(directory.path(), directory.path(), &mut config).unwrap();
        assert!(
            config
                .build
                .virtual_modules
                .iter()
                .any(|(specifier, _)| specifier == crate::rsc::CALL_SERVER_SPECIFIER)
        );
    }
}
