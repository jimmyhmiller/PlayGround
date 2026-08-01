//! Router-independent Next.js detection and compilation defaults.

use std::path::{Path, PathBuf};

use diffpack_core::transform::Target;

use crate::mdx::MdxConfig;
use crate::next_config::next_config_path;

pub fn is_next_project(root: &Path) -> bool {
    let manifest = root.join("package.json");
    if let Ok(text) = std::fs::read_to_string(&manifest) {
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(json) => {
                if ["dependencies", "devDependencies"]
                    .iter()
                    .any(|field| json.get(field).and_then(|deps| deps.get("next")).is_some())
                {
                    return true;
                }
            }
            Err(error) => eprintln!(
                "next detection: cannot parse {} ({error}); falling back to next.config detection",
                manifest.display(),
            ),
        }
    }
    next_config_path(root).is_some()
}

/// Router-independent App Router presence check used when choosing between
/// Next's two router integrations.
pub fn is_app_router(root: &Path) -> bool {
    if !is_next_project(root) {
        return false;
    }
    [root.join("app"), root.join("src/app")]
        .into_iter()
        .filter(|path| path.is_dir())
        .any(|path| contains_app_route(&path))
}

fn contains_app_route(directory: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && contains_app_route(&path) {
            return true;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((stem, extension)) = name.rsplit_once('.') else {
            continue;
        };
        if matches!(stem, "page" | "route") && matches!(extension, "js" | "jsx" | "ts" | "tsx") {
            return true;
        }
    }
    false
}

pub fn report_mdx_config(eval: Option<&serde_json::Value>) {
    let config = MdxConfig::from_eval(eval);
    if !config.configured {
        return;
    }
    let summary = config.summary();
    if config.unhonored_options().is_empty() {
        eprintln!(
            "[next.config] @next/mdx: {summary} — compiled by diffpack's native MDX compiler"
        );
    } else {
        eprintln!(
            "[next.config] @next/mdx: {summary} — .mdx/.md files are compiled with the app's own @mdx-js/mdx pipeline so these run"
        );
    }
}

pub fn process_browser_define(target: Target) -> &'static str {
    match target {
        Target::Client => "true",
        Target::Server | Target::IsolatedServer => "false",
    }
}

pub fn next_runtime_define(target: Target) -> &'static str {
    match target {
        Target::Server | Target::IsolatedServer => "\"nodejs\"",
        Target::Client => "\"\"",
    }
}

pub fn default_source_maps(
    target: Target,
    dev: bool,
    next_config: Option<&serde_json::Value>,
) -> bool {
    if dev {
        return true;
    }
    match target {
        Target::Client => production_browser_source_maps(next_config),
        Target::Server | Target::IsolatedServer => server_source_maps(next_config),
    }
}

pub fn server_source_maps(eval: Option<&serde_json::Value>) -> bool {
    eval.and_then(|value| value.get("serverSourceMaps"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub fn production_browser_source_maps(eval: Option<&serde_json::Value>) -> bool {
    eval.and_then(|value| value.get("productionBrowserSourceMaps"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub fn server_external_packages(eval: Option<&serde_json::Value>) -> Vec<String> {
    eval.and_then(|value| value.get("serverExternalPackages"))
        .and_then(|value| value.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

pub fn next_browser_polyfill_aliases(root: &Path) -> Vec<(String, String)> {
    const POLYFILLS: &[(&str, &str)] = &[
        ("assert", "assert"),
        ("buffer", "buffer"),
        ("constants", "constants-browserify"),
        ("crypto", "crypto-browserify"),
        ("domain", "domain-browser"),
        ("events", "events"),
        ("http", "stream-http"),
        ("https", "https-browserify"),
        ("os", "os-browserify"),
        ("path", "path-browserify"),
        ("process", "process"),
        ("punycode", "punycode"),
        ("querystring", "querystring-es3"),
        ("stream", "stream-browserify"),
        ("string_decoder", "string_decoder"),
        ("sys", "util"),
        ("timers", "timers-browserify"),
        ("tty", "tty-browserify"),
        ("url", "native-url"),
        ("util", "util"),
        ("vm", "vm-browserify"),
        ("zlib", "browserify-zlib"),
    ];
    let Some(compiled) = next_compiled_dir(root) else {
        return Vec::new();
    };
    let mut aliases = Vec::new();
    for (specifier, vendored) in POLYFILLS {
        let dir = compiled.join(vendored);
        if dir.is_dir() {
            let target = dir.to_string_lossy().into_owned();
            aliases.push(((*specifier).to_string(), target.clone()));
            aliases.push((format!("node:{specifier}"), target));
        }
    }
    aliases
}

fn next_compiled_dir(root: &Path) -> Option<PathBuf> {
    root.ancestors()
        .map(|dir| dir.join("node_modules/next"))
        .find(|next| next.join("package.json").is_file())
        .map(|next| next.join("dist/compiled"))
        .filter(|compiled| compiled.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_file_or_dependency_identifies_a_next_project() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_next_project(dir.path()));
        std::fs::write(dir.path().join("next.config.mjs"), "export default {}").unwrap();
        assert!(is_next_project(dir.path()));
    }

    #[test]
    fn target_defines_and_map_defaults_match_next() {
        assert_eq!(process_browser_define(Target::Client), "true");
        assert_eq!(next_runtime_define(Target::Server), "\"nodejs\"");
        assert!(!default_source_maps(Target::Server, false, None));
        assert!(!default_source_maps(Target::Client, false, None));
        let server_maps = serde_json::json!({ "serverSourceMaps": true });
        assert!(default_source_maps(
            Target::Server,
            false,
            Some(&server_maps)
        ));
    }
}
