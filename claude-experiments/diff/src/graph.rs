use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use crate::parser::parse_dependencies;

pub type ModuleId = String;

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct GraphSnapshot {
    pub entry: ModuleId,
    pub modules: BTreeMap<ModuleId, u64>,
    pub edges: BTreeSet<(ModuleId, ModuleId)>,
    pub diagnostics: Vec<String>,
}

pub fn scan_graph(entry: &Path) -> Result<GraphSnapshot, String> {
    let entry = entry
        .canonicalize()
        .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
    let root = entry
        .parent()
        .ok_or_else(|| format!("entry has no parent: {}", entry.display()))?
        .to_path_buf();
    let entry_id = module_id(&root, &entry);
    let mut snapshot = GraphSnapshot {
        entry: entry_id,
        ..GraphSnapshot::default()
    };
    let mut queue = VecDeque::from([entry]);
    let mut visited = BTreeSet::new();

    while let Some(path) = queue.pop_front() {
        if !visited.insert(path.clone()) {
            continue;
        }

        let id = module_id(&root, &path);
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) => {
                snapshot
                    .diagnostics
                    .push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        snapshot
            .modules
            .insert(id.clone(), content_hash(source.as_bytes()));

        let parsed = parse_dependencies(&path, &source);
        snapshot.diagnostics.extend(
            parsed
                .errors
                .into_iter()
                .map(|error| format!("{id}: {error}")),
        );

        for specifier in parsed.dependencies {
            if !specifier.starts_with('.') {
                snapshot.diagnostics.push(format!(
                    "{id}: bare import {specifier:?} is outside the PoC resolver"
                ));
                continue;
            }

            match resolve_relative(&path, &specifier) {
                Some(target) => {
                    let target_id = module_id(&root, &target);
                    snapshot.edges.insert((id.clone(), target_id));
                    queue.push_back(target);
                }
                None => snapshot
                    .diagnostics
                    .push(format!("{id}: cannot resolve {specifier:?}")),
            }
        }
    }

    snapshot.diagnostics.sort();
    Ok(snapshot)
}

fn resolve_relative(importer: &Path, specifier: &str) -> Option<PathBuf> {
    let parent = importer.parent()?;
    let candidate = parent.join(specifier);
    let mut candidates = vec![candidate.clone()];

    if candidate.extension().is_none() {
        for extension in ["js", "jsx", "ts", "tsx", "mjs", "cjs"] {
            candidates.push(candidate.with_extension(extension));
        }
        for extension in ["js", "jsx", "ts", "tsx", "mjs", "cjs"] {
            candidates.push(candidate.join(format!("index.{extension}")));
        }
    }

    candidates
        .into_iter()
        .find(|path| path.is_file())
        .and_then(|path| path.canonicalize().ok())
}

fn module_id(root: &Path, path: &Path) -> ModuleId {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn content_hash(bytes: &[u8]) -> u64 {
    // FNV-1a is sufficient here: this hash is an invalidation key, not a security boundary.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn scans_only_transitively_reachable_relative_modules() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("entry.js"), "import './a.js';").unwrap();
        fs::write(directory.path().join("a.js"), "export { b } from './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 1;").unwrap();
        fs::write(directory.path().join("unused.js"), "export const nope = 1;").unwrap();

        let graph = scan_graph(&directory.path().join("entry.js")).unwrap();

        assert_eq!(graph.modules.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert!(!graph.modules.contains_key("unused.js"));
        assert!(graph.diagnostics.is_empty(), "{:?}", graph.diagnostics);
    }
}
