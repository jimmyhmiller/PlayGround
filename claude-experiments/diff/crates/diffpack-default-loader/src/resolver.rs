//! Construction of the paired ESM/CommonJS Node resolvers.

use std::path::{Path, PathBuf};

use oxc_resolver::{ResolveOptions, Resolver, TsconfigDiscovery};

const ESM_ONLY_CONDITIONS: [&str; 2] = ["import", "module"];

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ImportSyntax {
    Esm,
    CommonJs,
}

#[derive(Debug, Clone, Default)]
pub struct ResolverConfig {
    pub conditions: Vec<String>,
    pub main_fields: Vec<String>,
    pub browser: bool,
}

pub struct Resolvers {
    esm: Resolver,
    common_js: Resolver,
}

impl Resolvers {
    pub fn new(config: &ResolverConfig) -> Self {
        Self {
            esm: Resolver::new(options(config, ImportSyntax::Esm)),
            common_js: Resolver::new(options(config, ImportSyntax::CommonJs)),
        }
    }

    pub fn for_syntax(&self, syntax: ImportSyntax) -> &Resolver {
        match syntax {
            ImportSyntax::Esm => &self.esm,
            ImportSyntax::CommonJs => &self.common_js,
        }
    }
}

impl std::ops::Deref for Resolvers {
    type Target = Resolver;

    fn deref(&self) -> &Resolver {
        &self.esm
    }
}

fn options(config: &ResolverConfig, syntax: ImportSyntax) -> ResolveOptions {
    let condition_names = match syntax {
        ImportSyntax::Esm if config.conditions.is_empty() => {
            vec!["import".into(), "module".into(), "default".into()]
        }
        ImportSyntax::Esm => {
            let mut names = config.conditions.clone();
            for fallback in ["import", "default"] {
                if !names.iter().any(|name| name == fallback) {
                    names.push(fallback.to_string());
                }
            }
            names
        }
        ImportSyntax::CommonJs => {
            let mut names = config
                .conditions
                .iter()
                .filter(|name| !ESM_ONLY_CONDITIONS.contains(&name.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            for fallback in ["require", "default"] {
                if !names.iter().any(|name| name == fallback) {
                    names.push(fallback.to_string());
                }
            }
            names
        }
    };
    ResolveOptions {
        tsconfig: Some(TsconfigDiscovery::Auto),
        extensions: [
            ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".mdx", ".md",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        extension_alias: vec![
            (
                ".js".into(),
                vec![".ts".into(), ".tsx".into(), ".js".into(), ".jsx".into()],
            ),
            (".mjs".into(), vec![".mts".into(), ".mjs".into()]),
            (".cjs".into(), vec![".cts".into(), ".cjs".into()]),
        ],
        condition_names,
        alias_fields: if config.browser {
            vec![vec!["browser".into()]]
        } else {
            Vec::new()
        },
        main_fields: if !config.main_fields.is_empty() {
            config.main_fields.clone()
        } else if config.browser {
            vec!["browser".into(), "module".into(), "main".into()]
        } else {
            vec!["module".into(), "main".into()]
        },
        ..ResolveOptions::default()
    }
}

/// Resolves transform-discovered worker entry specifiers relative to their
/// importing module.
pub fn resolve_worker_entries(
    resolver: &Resolver,
    importer: &Path,
    workers: &[(String, String)],
) -> Result<Vec<(String, PathBuf)>, String> {
    workers
        .iter()
        .map(|(key, specifier)| {
            resolver
                .resolve_file(importer, specifier)
                .map(|resolution| (key.clone(), resolution.full_path().to_path_buf()))
                .map_err(|error| {
                    format!(
                        "cannot resolve worker entry {specifier:?} from {}: {error}",
                        importer.display()
                    )
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commonjs_replaces_esm_conditions_but_keeps_environment_conditions() {
        let config = ResolverConfig {
            conditions: vec!["node".into(), "import".into(), "production".into()],
            ..Default::default()
        };
        let options = options(&config, ImportSyntax::CommonJs);
        assert_eq!(
            options.condition_names,
            ["node", "production", "require", "default"]
        );
    }
}
