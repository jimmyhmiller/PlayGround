//! Discovery and best-effort evaluation of `next.config.*`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const NEXT_CONFIG_EXTS: [&str; 6] = ["js", "mjs", "cjs", "ts", "mts", "cts"];
static EVAL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn next_config_path(root: &Path) -> Option<PathBuf> {
    NEXT_CONFIG_EXTS
        .iter()
        .map(|ext| root.join(format!("next.config.{ext}")))
        .find(|path| path.is_file())
}

pub fn run_next_config_eval(root: &Path) -> Option<serde_json::Value> {
    let config = next_config_path(root)?;
    let loader = std::env::temp_dir().join("diffpack-next-config-eval.mjs");
    let loader_bytes = include_str!("next_config/evaluator.mjs");
    if std::fs::read_to_string(&loader).ok().as_deref() != Some(loader_bytes) {
        let staged = std::env::temp_dir().join(format!(
            "diffpack-next-config-eval-{}.mjs",
            std::process::id(),
        ));
        std::fs::write(&staged, loader_bytes).ok()?;
        std::fs::rename(&staged, &loader).ok()?;
    }
    let payload = std::env::temp_dir().join(format!(
        "diffpack-next-config-{}-{}.json",
        std::process::id(),
        EVAL_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    ));
    let _ = std::fs::remove_file(&payload);
    let out = std::process::Command::new("node")
        .arg(&loader)
        .arg(&config)
        .arg(&payload)
        .current_dir(root)
        .output()
        .ok()?;
    if !out.stderr.is_empty() {
        eprintln!(
            "[next.config] {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = std::fs::read_to_string(&payload).ok();
    let _ = std::fs::remove_file(&payload);
    if !out.status.success() {
        return None;
    }
    serde_json::from_str(&text?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_every_supported_config_extension_in_priority_order() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(next_config_path(dir.path()), None);
        std::fs::write(dir.path().join("next.config.ts"), "export default {}").unwrap();
        assert_eq!(
            next_config_path(dir.path()),
            Some(dir.path().join("next.config.ts"))
        );
        std::fs::write(dir.path().join("next.config.js"), "module.exports = {}").unwrap();
        assert_eq!(
            next_config_path(dir.path()),
            Some(dir.path().join("next.config.js"))
        );
    }
}
