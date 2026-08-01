//! Vite-compatible pre-parse source policy for the filesystem graph driver.

use std::borrow::Cow;
use std::path::Path;

use diffpack_core::transform::Target;
use diffpack_default_loader::source_policy::SourceIntegrationPolicy;

#[derive(Debug, Default, Clone)]
pub struct ViteSourcePolicy {
    pub import_meta_env: Option<crate::import_meta_env::ImportMetaEnv>,
    pub import_meta_glob: Option<crate::import_meta_glob::ImportMetaGlob>,
    pub defines: Vec<(String, String)>,
}

impl SourceIntegrationPolicy for ViteSourcePolicy {
    fn transform(
        &self,
        path: &Path,
        source: &str,
        target: Target,
    ) -> Result<Option<String>, String> {
        let mut current = Cow::Borrowed(source);
        if let Some(options) = self.import_meta_glob.as_ref()
            && let Some(rewritten) = crate::import_meta_glob::transform(path, &current, options)?
        {
            current = Cow::Owned(rewritten);
        }
        if let Some(options) = self.import_meta_env.as_ref()
            && let Some(rewritten) =
                crate::import_meta_env::transform(path, &current, options, target == Target::Server)
        {
            current = Cow::Owned(rewritten);
        }
        if !self.defines.is_empty()
            && let Some(rewritten) = crate::vite_define::transform(path, &current, &self.defines)
        {
            current = Cow::Owned(rewritten);
        }
        if !matches!(current, Cow::Borrowed(_))
            && std::env::var_os("DIFFPACK_DISABLE_DEAD_BRANCH").is_none()
            && let Some(rewritten) = diffpack_core::dead_branch::transform(path, &current)
        {
            current = Cow::Owned(rewritten);
        }
        Ok(match current {
            Cow::Borrowed(_) => None,
            Cow::Owned(rewritten) => Some(rewritten),
        })
    }

    fn development(&self) -> Option<std::sync::Arc<dyn SourceIntegrationPolicy>> {
        let mut policy = self.clone();
        const NODE_ENV: &str = "process.env.NODE_ENV";
        let value = "\"development\"".to_string();
        match policy.defines.iter_mut().find(|(key, _)| key == NODE_ENV) {
            Some(existing) => existing.1 = value,
            None => policy.defines.push((NODE_ENV.to_string(), value)),
        }
        if let Some(env) = policy.import_meta_env.as_mut() {
            env.mode = "development".to_string();
        }
        Some(std::sync::Arc::new(policy))
    }

    fn defines(&self) -> &[(String, String)] {
        &self.defines
    }
}
