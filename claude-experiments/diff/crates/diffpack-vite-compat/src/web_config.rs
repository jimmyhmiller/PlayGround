//! Adapter from resolved Vite conventions to Diffpack's neutral Web profile.

use std::path::{Path, PathBuf};

use diffpack_web::config::{EmittedPage, WebConfig, set_node_env};

#[derive(Debug, Clone)]
pub struct ViteWebProfile {
    pub web: WebConfig,
    pub manifest_name: Option<String>,
    pub optimize_deps_exclude: Vec<String>,
    pub copy_public_dir: bool,
}

impl ViteWebProfile {
    /// Writes the optional Vite build manifest from Web-owned emission records.
    /// Returns the configured manifest name for user-facing build reporting.
    pub fn write_manifest(
        &self,
        output_dir: &Path,
        pages: &[EmittedPage],
    ) -> Result<Option<&str>, String> {
        let Some(name) = self.manifest_name.as_deref() else {
            return Ok(None);
        };
        let path = output_dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::write(&path, crate::vite_manifest::render(pages))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        Ok(Some(name))
    }
}

/// Constructs a Web build profile with Vite configuration and source semantics.
pub fn derive(root: &Path) -> Result<ViteWebProfile, String> {
    let mut config = diffpack_web::config::derive_web_config(root)?;
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let root = root.as_path();
    let resolved = match crate::vite_config::resolve(root, "production") {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!(
                "warning: could not evaluate vite config ({error}); continuing with defaults"
            );
            None
        }
    };
    let base = resolved
        .as_ref()
        .and_then(|resolved| resolved.base.clone())
        .unwrap_or_else(|| "/".to_string());
    let base = if base.ends_with('/') {
        base
    } else {
        format!("{base}/")
    };

    config.build.scss.additional_data = resolved
        .as_ref()
        .and_then(|resolved| resolved.scss_additional_data.clone());
    config.build.jsx = resolved
        .as_ref()
        .map(|resolved| resolved.jsx.clone())
        .unwrap_or_default();
    let mut defines = resolved
        .as_ref()
        .map(|resolved| resolved.define.clone())
        .unwrap_or_default();
    set_node_env(&mut defines, "production");

    config.build.conditions.push("production".to_string());
    if let Some(extra) = resolved
        .as_ref()
        .map(|resolved| &resolved.resolve_conditions)
    {
        for condition in extra {
            if !config.build.conditions.contains(condition) {
                config.build.conditions.push(condition.clone());
            }
        }
    }
    if let Some(main_fields) = resolved
        .as_ref()
        .map(|resolved| resolved.main_fields.clone())
    {
        config.build.main_fields = main_fields;
    }
    config.build.base = base.clone();
    config.build.asset_inline_limit = 4096;
    config.build.aliases = resolved
        .as_ref()
        .map(|resolved| resolved.alias.clone())
        .unwrap_or_default()
        .into_iter()
        .map(|(find, replacement)| {
            let replacement = if let Some(rest) = replacement.strip_prefix('/')
                && !Path::new(&replacement).exists()
                && root.join(rest).exists()
            {
                root.join(rest).to_string_lossy().into_owned()
            } else {
                replacement
            };
            (find, replacement)
        })
        .collect();

    let vite_vars = crate::env_file::load_vite_env(root, "production")?;
    config.build.source_policy = std::sync::Arc::new(crate::source_policy::ViteSourcePolicy {
        import_meta_env: Some(crate::import_meta_env::ImportMetaEnv {
            base: base.clone(),
            mode: "production".to_string(),
            vite_vars,
        }),
        import_meta_glob: Some(crate::import_meta_glob::ImportMetaGlob {
            root: root.to_path_buf(),
        }),
        defines,
    });

    if let Some(resolved) = resolved.as_ref() {
        for package in &resolved.dedupe {
            if config.build.aliases.iter().any(|(find, _)| find == package) {
                continue;
            }
            let target = root.join("node_modules").join(package);
            if target.is_dir() {
                config
                    .build
                    .aliases
                    .push((package.clone(), target.to_string_lossy().into_owned()));
            }
        }
        config.inputs = resolved
            .inputs
            .iter()
            .map(|(name, path)| (name.clone(), PathBuf::from(path)))
            .collect();
        config.proxy = resolved.proxy.clone();
        config.out_dir = resolved.out_dir.as_ref().map(|out_dir| root.join(out_dir));
        if let Some(assets_dir) = resolved.assets_dir.as_deref()
            && assets_dir.trim_matches('/') != "assets"
        {
            return Err(format!(
                "vite config sets build.assetsDir = {assets_dir:?}, which diffpack does not \
                 implement: it emits hashed assets to `assets/` unconditionally, so every \
                 asset URL in the build would be wrong. Remove build.assetsDir (or set it to \
                 \"assets\") to build this project with diffpack."
            ));
        }
    }
    config.configuration_files = crate::vite_config::config_file(root).into_iter().collect();
    config.base = base;
    let manifest_name = resolved.as_ref().and_then(|resolved| {
        resolved.manifest.then(|| {
            resolved
                .manifest_name
                .clone()
                .unwrap_or_else(|| ".vite/manifest.json".into())
        })
    });
    let optimize_deps_exclude = resolved
        .as_ref()
        .map(|resolved| resolved.optimize_deps_exclude.clone())
        .unwrap_or_default();
    Ok(ViteWebProfile {
        web: config,
        manifest_name,
        optimize_deps_exclude,
        copy_public_dir: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffpack_core::runtime::{RuntimeIntegrationPolicy, RuntimePolicyRequest};
    use std::process::Command;

    #[test]
    fn build_out_dir_is_resolved_against_the_project_root() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(
            root.join("vite.config.mjs"),
            "export default { build: { outDir: 'build' } };",
        )
        .unwrap();
        assert_eq!(
            derive(root).unwrap().web.out_dir,
            Some(root.canonicalize().unwrap().join("build"))
        );
        std::fs::write(root.join("vite.config.mjs"), "export default {};").unwrap();
        assert_eq!(derive(root).unwrap().web.out_dir, None);
    }

    #[test]
    fn non_default_assets_dir_is_a_named_error() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(
            root.join("vite.config.mjs"),
            "export default { build: { assetsDir: 'static' } };",
        )
        .unwrap();
        let error = derive(root).unwrap_err();
        assert!(error.contains("build.assetsDir"), "{error}");
        assert!(error.contains("\"static\""), "{error}");
    }

    #[test]
    fn vite_runtime_profile_is_the_neutral_web_runtime() {
        let output = diffpack_web::policies::WebRuntimePolicy
            .configure(RuntimePolicyRequest {
                format: diffpack_core::ModuleFormat::BrowserEsm,
                is_main: true,
                hmr: false,
                entry_id: "entry",
                entry_runtime_id: 0,
                any_async: false,
                base: "/",
                chunk_files: &[],
                modules: &[],
                browser_process_shim: false,
            })
            .unwrap();
        assert_eq!(output.describe(), ["browser-require-native@diffpack-web"]);
    }
}
