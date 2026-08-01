//! Vite's `.env` filename precedence and client exposure policy.

use std::path::Path;

pub use diffpack_default_loader::env_file::parse;

pub fn load_vite_env(root: &Path, mode: &str) -> Result<Vec<(String, String)>, String> {
    let files = [
        ".env".to_string(),
        ".env.local".to_string(),
        format!(".env.{mode}"),
        format!(".env.{mode}.local"),
    ];
    let mut merged = diffpack_default_loader::env_file::load_files(root, &files)?;
    for (name, value) in std::env::vars().filter(|(name, _)| name.starts_with("VITE_")) {
        match merged.iter_mut().find(|(existing, _)| existing == &name) {
            Some(entry) => entry.1 = value,
            None => merged.push((name, value)),
        }
    }
    Ok(merged
        .into_iter()
        .filter(|(name, _)| name.starts_with("VITE_"))
        .collect())
}
