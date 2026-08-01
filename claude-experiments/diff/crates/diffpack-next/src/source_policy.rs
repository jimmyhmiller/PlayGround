//! Next-owned source transforms assembled from framework-neutral mechanisms.

use std::path::Path;

use diffpack_core::transform::Target;
use diffpack_default_loader::source_policy::SourceIntegrationPolicy;

#[derive(Debug, Default, Clone)]
pub struct NextSourcePolicy {
    pub defines: Vec<(String, String)>,
    pub external_singletons: Vec<std::path::PathBuf>,
    pub external_named_singletons: Vec<(std::path::PathBuf, Vec<String>)>,
    pub webpack_runtime_singletons: Vec<std::path::PathBuf>,
}

impl SourceIntegrationPolicy for NextSourcePolicy {
    fn transform(
        &self,
        path: &Path,
        source: &str,
        _target: Target,
    ) -> Result<Option<String>, String> {
        let canonical_path = path.canonicalize().ok();
        if self.webpack_runtime_singletons.iter().any(|candidate| {
            candidate == path
                || canonical_path
                    .as_ref()
                    .is_some_and(|path| candidate == path)
        }) {
            let target = serde_json::to_string(&path.to_string_lossy()).unwrap();
            return Ok(Some(format!(
                "const __diffpack_old_turbopack = process.env.TURBOPACK;\ndelete process.env.TURBOPACK;\ntry {{ module.exports = process.getBuiltinModule('module').createRequire(process.cwd() + '/package.json')({target}); }} finally {{ if (__diffpack_old_turbopack === undefined) delete process.env.TURBOPACK; else process.env.TURBOPACK = __diffpack_old_turbopack; }}"
            )));
        }
        if let Some((_, names)) = self
            .external_named_singletons
            .iter()
            .find(|(candidate, _)| {
                candidate == path
                    || canonical_path
                        .as_ref()
                        .is_some_and(|path| candidate == path)
            })
        {
            let target = serde_json::to_string(&path.to_string_lossy()).unwrap();
            let mut wrapper = format!(
                "const __diffpack_native = process.getBuiltinModule('module').createRequire(process.cwd() + '/package.json')({target});\n"
            );
            for name in names {
                let name = serde_json::to_string(name).unwrap();
                wrapper.push_str(&format!("exports[{name}] = __diffpack_native[{name}];\n"));
            }
            return Ok(Some(wrapper));
        }
        if self.external_singletons.iter().any(|candidate| {
            candidate == path
                || canonical_path
                    .as_ref()
                    .is_some_and(|path| candidate == path)
        }) {
            return Ok(Some(format!(
                "module.exports = process.getBuiltinModule('module').createRequire(process.cwd() + '/package.json')({});",
                serde_json::to_string(&path.to_string_lossy()).unwrap()
            )));
        }
        let Some(mut rewritten) =
            diffpack_default_loader::define::transform(path, source, &self.defines)
        else {
            return Ok(None);
        };
        if std::env::var_os("DIFFPACK_DISABLE_DEAD_BRANCH").is_none()
            && let Some(folded) = diffpack_core::dead_branch::transform(path, &rewritten)
        {
            rewritten = folded;
        }
        Ok(Some(rewritten))
    }

    fn development(&self) -> Option<std::sync::Arc<dyn SourceIntegrationPolicy>> {
        let mut policy = self.clone();
        const NODE_ENV: &str = "process.env.NODE_ENV";
        let value = "\"development\"".to_string();
        match policy.defines.iter_mut().find(|(key, _)| key == NODE_ENV) {
            Some(existing) => existing.1 = value,
            None => policy.defines.push((NODE_ENV.to_string(), value)),
        }
        Some(std::sync::Arc::new(policy))
    }

    fn defines(&self) -> &[(String, String)] {
        &self.defines
    }
}
