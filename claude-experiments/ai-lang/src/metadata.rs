//! Per-hash metadata store.
//!
//! Each def hash can have an optional JSON metadata file at
//! `.ai-lang/meta/<hex>.json`. The top-level keys are system-owned;
//! user-defined keys go under the `meta` key.
//!
//! Schema:
//! ```json
//! {
//!   "tags": ["ai:test", "my-category"],
//!   "doc": "optional description",
//!   "meta": { "anything": "user-defined" }
//! }
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::codebase::Codebase;
use crate::hash::Hash;
use crate::jsonl::Json;

fn meta_path(root: &std::path::Path, hash: &Hash) -> PathBuf {
    root.join("meta").join(format!("{}.json", hash.to_hex()))
}

/// Load metadata for a hash, or `None` if it doesn't exist or can't be parsed.
pub fn load(cb: &Codebase, hash: &Hash) -> Option<Json> {
    let path = meta_path(cb.root(), hash);
    let text = std::fs::read_to_string(&path).ok()?;
    crate::jsonl::parse(&text).ok()
}

/// Save metadata for a hash. Creates the `meta/` directory if needed.
pub fn save(cb: &Codebase, hash: &Hash, json: &Json) -> Result<(), std::io::Error> {
    let path = meta_path(cb.root(), hash);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, json.to_string())
}

/// Remove the metadata file for a hash (used when untagging the last tag).
pub fn delete(cb: &Codebase, hash: &Hash) -> Result<(), std::io::Error> {
    let path = meta_path(cb.root(), hash);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Get or create a mutable Json object for a hash.
fn load_or_empty(cb: &Codebase, hash: &Hash) -> Json {
    load(cb, hash).unwrap_or_else(|| Json::Object(BTreeMap::new()))
}

/// Add a tag. Idempotent.
pub fn add_tag(cb: &Codebase, hash: &Hash, tag: &str) -> Result<(), String> {
    let mut json = load_or_empty(cb, hash);
    let tags = match &mut json {
        Json::Object(map) => {
            map.entry("tags".to_string())
                .or_insert_with(|| Json::Array(Vec::new()))
        }
        _ => return Err("metadata is not an object".to_string()),
    };
    if let Json::Array(arr) = tags {
        if !arr.iter().any(|j| j.as_str() == Some(tag)) {
            arr.push(Json::Str(tag.to_string()));
        }
    } else {
        *tags = Json::Array(vec![Json::Str(tag.to_string())]);
    }
    save(cb, hash, &json).map_err(|e| e.to_string())
}

/// Remove a tag.
pub fn remove_tag(cb: &Codebase, hash: &Hash, tag: &str) -> Result<(), String> {
    let mut json = match load(cb, hash) {
        Some(j) => j,
        None => return Ok(()), // nothing to remove
    };
    let is_empty = match &mut json {
        Json::Object(map) => {
            if let Some(Json::Array(arr)) = map.get_mut("tags") {
                arr.retain(|j| j.as_str() != Some(tag));
                arr.is_empty() && map.get("doc").is_none() && map.get("meta").is_none()
            } else {
                true
            }
        }
        _ => true,
    };
    if is_empty {
        delete(cb, hash).map_err(|e| e.to_string())
    } else {
        save(cb, hash, &json).map_err(|e| e.to_string())
    }
}

/// Set a user-defined key under `meta`.
pub fn set_meta(cb: &Codebase, hash: &Hash, key: &str, value: &Json) -> Result<(), String> {
    let mut json = load_or_empty(cb, hash);
    match &mut json {
        Json::Object(map) => {
            let meta = map.entry("meta".to_string())
                .or_insert_with(|| Json::Object(BTreeMap::new()));
            if let Json::Object(meta_map) = meta {
                meta_map.insert(key.to_string(), value.clone());
            } else {
                *meta = {
                    let mut m = BTreeMap::new();
                    m.insert(key.to_string(), value.clone());
                    Json::Object(m)
                };
            }
        }
        _ => return Err("metadata is not an object".to_string()),
    }
    save(cb, hash, &json).map_err(|e| e.to_string())
}

/// Remove a user-defined key from `meta`.
pub fn unset_meta(cb: &Codebase, hash: &Hash, key: &str) -> Result<(), String> {
    let mut json = match load(cb, hash) {
        Some(j) => j,
        None => return Ok(()),
    };
    match &mut json {
        Json::Object(map) => {
            if let Some(Json::Object(meta_map)) = map.get_mut("meta") {
                meta_map.remove(key);
                if meta_map.is_empty() {
                    map.remove("meta");
                }
            }
        }
        _ => {}
    }
    save(cb, hash, &json).map_err(|e| e.to_string())
}

/// Set the top-level `doc` key.
pub fn set_doc(cb: &Codebase, hash: &Hash, doc: &str) -> Result<(), String> {
    let mut json = load_or_empty(cb, hash);
    match &mut json {
        Json::Object(map) => {
            map.insert("doc".to_string(), Json::Str(doc.to_string()));
        }
        _ => return Err("metadata is not an object".to_string()),
    }
    save(cb, hash, &json).map_err(|e| e.to_string())
}

/// Check if a hash has a specific tag.
pub fn has_tag(cb: &Codebase, hash: &Hash, tag: &str) -> bool {
    match load(cb, hash) {
        Some(Json::Object(ref map)) => {
            map.get("tags")
                .and_then(|t| t.as_array())
                .map(|arr| arr.iter().any(|j| j.as_str() == Some(tag)))
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// Return all hashes in the meta directory that have a specific tag.
pub fn hashes_with_tag(
    root: &std::path::Path,
    tag: &str,
) -> Result<Vec<Hash>, std::io::Error> {
    let meta_dir = root.join("meta");
    let mut result = Vec::new();
    let entries = match std::fs::read_dir(&meta_dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(result),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.ends_with(".json") {
            continue;
        }
        let hex = &name_str[..name_str.len() - 5];
        let hash = match crate::codebase::parse_hex_hash(hex) {
            Some(h) => h,
            None => continue,
        };
        // Quick scan: read the file and check for the tag string.
        // Avoid full JSON parse for the common case (most files tagged).
        let text = std::fs::read_to_string(entry.path())?;
        if text.contains(&format!("\"{}\"", tag)) {
            result.push(hash);
        }
    }
    Ok(result)
}

/// Copy metadata from old hash to new hash (used on same-type update).
/// Does nothing if the old hash has no metadata.
pub fn copy_meta(cb: &Codebase, old_hash: &Hash, new_hash: &Hash) -> Result<(), String> {
    let json = match load(cb, old_hash) {
        Some(j) => j,
        None => return Ok(()),
    };
    save(cb, new_hash, &json).map_err(|e| e.to_string())
}
