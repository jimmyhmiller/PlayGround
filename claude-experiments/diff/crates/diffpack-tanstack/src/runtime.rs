//! Native TanStack Start Node server runtime emission.

use std::fs;
use std::path::{Path, PathBuf};

use diffpack_default_loader::output::write_if_changed;

const FILES: &[(&str, &str)] = &[
    ("index.mjs", include_str!("server_runtime/index.mjs")),
    (
        "_ssr/node-adapter.mjs",
        include_str!("server_runtime/_ssr/node-adapter.mjs"),
    ),
    ("_ssr/ssr.mjs", include_str!("server_runtime/_ssr/ssr.mjs")),
    (
        "_ssr/router.mjs",
        include_str!("server_runtime/_ssr/router.mjs"),
    ),
];

const HMR_SSR_ENTRY: &str = r#"import serverEntry from "../server.mjs";

export function resolveFetch(entry) {
  const seen = new Set();
  const queue = [entry];
  while (queue.length > 0) {
    const candidate = queue.shift();
    if (candidate == null || seen.has(candidate)) continue;
    seen.add(candidate);
    if (typeof candidate === "function") return candidate;
    if (typeof candidate.fetch === "function") return candidate.fetch.bind(candidate);
    if (typeof candidate === "object") queue.push(candidate.default);
  }
  throw new Error("diffpack ssr: ./server.mjs default export exposes no fetch handler");
}

export const fetch = (request) =>
  resolveFetch(globalThis.__diffpack_ssr_entry || serverEntry)(request);
export default { fetch };
"#;

/// Entry initialization required by TanStack Start before bundled modules run.
pub const SERVER_ENTRY_ENVIRONMENT_PRELUDE: &str =
    "process.env.TSS_SERVER_FN_BASE ??= \"/_serverFn/\";\n";

/// TanStack-specific browser environment layered on top of web compatibility.
pub const BROWSER_ENTRY_ENVIRONMENT_PRELUDE: &str = "globalThis.process.env.TSS_SERVER_FN_BASE=globalThis.process.env.TSS_SERVER_FN_BASE||\"/_serverFn/\";\n";

/// Writes the fixed server bootstrap, Node adapter, SSR entry, and route-manifest
/// bridge beside an emitted TanStack server graph.
pub fn write_server_entry(server_dir: &Path, hmr: bool) -> Result<Vec<PathBuf>, String> {
    let mut written = Vec::with_capacity(FILES.len());
    for (relative, source) in FILES {
        let path = server_dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        let source = if hmr && *relative == "_ssr/ssr.mjs" {
            HMR_SSR_ENTRY
        } else {
            source
        };
        write_if_changed(&path, source.as_bytes())?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_the_complete_runtime_and_dev_ssr_variant() {
        let directory = tempfile::tempdir().unwrap();
        let written = write_server_entry(directory.path(), true).unwrap();
        assert_eq!(written.len(), 4);
        assert!(directory.path().join("index.mjs").is_file());
        assert!(directory.path().join("_ssr/router.mjs").is_file());
        let ssr = fs::read_to_string(directory.path().join("_ssr/ssr.mjs")).unwrap();
        assert!(ssr.contains("globalThis.__diffpack_ssr_entry"));
    }
}
