//! Framework-neutral filesystem event classification and coalescing for dev servers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant, SystemTime};

use notify::{RecursiveMode, Watcher};

static EVENTS_SEEN: AtomicU64 = AtomicU64::new(0);
static WATCH_BASE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the tree relative to which generated/dependency directories are ignored.
pub fn set_base(base: &Path) {
    *WATCH_BASE.lock().unwrap() = Some(base.to_path_buf());
}

/// Count an event when at least one path is an actionable source module.
pub fn note_paths(paths: &[PathBuf]) {
    let base = WATCH_BASE.lock().unwrap().clone();
    if paths.iter().any(|path| {
        is_module_path(path)
            && base
                .as_deref()
                .is_none_or(|base| !is_dependency_or_generated(path, base))
    }) {
        EVENTS_SEEN.fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot used to cancel deferred work when a newer edit arrives.
#[derive(Copy, Clone)]
pub struct EventEpoch(u64);

impl EventEpoch {
    pub fn now() -> Self {
        Self(EVENTS_SEEN.load(Ordering::Relaxed))
    }

    pub fn superseded(&self) -> bool {
        EVENTS_SEEN.load(Ordering::Relaxed) != self.0
    }
}

pub type WatchReceiver = Receiver<notify::Result<notify::Event>>;
pub type WatchRoot = (PathBuf, RecursiveMode);

pub fn start(root: &Path) -> Result<WatchReceiver, String> {
    start_paths(&[(root.to_path_buf(), RecursiveMode::Recursive)])
}

pub fn start_paths(roots: &[WatchRoot]) -> Result<WatchReceiver, String> {
    let (events, receiver) = mpsc::channel();
    start_paths_into(roots, events)?;
    Ok(receiver)
}

pub fn start_paths_into(
    roots: &[WatchRoot],
    events: Sender<notify::Result<notify::Event>>,
) -> Result<(), String> {
    let mut watcher = notify::recommended_watcher({
        let events = events.clone();
        move |event: notify::Result<notify::Event>| {
            if let Ok(event) = &event {
                note_paths(&event.paths);
            }
            let _ = events.send(event);
        }
    })
    .map_err(|error| format!("cannot create filesystem watcher: {error}"))?;
    for (path, mode) in roots {
        watcher.watch(path, *mode).map_err(|error| {
            format!(
                "cannot start filesystem watcher on {}: {error}",
                path.display()
            )
        })?;
    }
    Box::leak(Box::new(watcher));
    spawn_supplement_poller(roots.to_vec(), events);
    Ok(())
}

pub fn uncovered_roots(wanted: &[WatchRoot], covered: &[WatchRoot]) -> Vec<WatchRoot> {
    wanted
        .iter()
        .filter(|(root, mode)| {
            !covered.iter().any(|(existing, existing_mode)| {
                existing == root
                    || (*existing_mode == RecursiveMode::Recursive && root.starts_with(existing))
                    || (*mode == RecursiveMode::NonRecursive && root.starts_with(existing))
            })
        })
        .cloned()
        .collect()
}

fn spawn_supplement_poller(roots: Vec<WatchRoot>, events: Sender<notify::Result<notify::Event>>) {
    let _ = std::thread::Builder::new()
        .name("diffpack-fast-poll".into())
        .spawn(move || {
            let mut snapshot = HashMap::new();
            let mut first = true;
            loop {
                let scan_started = Instant::now();
                let mut current = HashMap::new();
                for (root, mode) in &roots {
                    scan_root(root, *mode, &mut current);
                }
                if !first {
                    for (path, signature) in &current {
                        if snapshot.get(path) != Some(signature) {
                            let event = notify::Event::new(notify::EventKind::Modify(
                                notify::event::ModifyKind::Any,
                            ))
                            .add_path(path.clone());
                            note_paths(&event.paths);
                            if events.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                    }
                }
                snapshot = current;
                first = false;
                let scan = scan_started.elapsed();
                std::thread::sleep(std::cmp::max(Duration::from_millis(2), scan * 4));
            }
        });
}

fn scan_root(root: &Path, mode: RecursiveMode, out: &mut HashMap<PathBuf, (SystemTime, u64)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            if mode == RecursiveMode::Recursive {
                let name = entry.file_name();
                if name == "node_modules" || name.as_encoded_bytes().starts_with(b".") {
                    continue;
                }
                scan_root(&path, mode, out);
            }
        } else if let Ok(meta) = entry.metadata() {
            out.insert(
                path,
                (
                    meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    meta.len(),
                ),
            );
        }
    }
}

/// Whether a path is under an installed dependency or generated/cache directory.
pub fn is_dependency_or_generated(path: &Path, base: &Path) -> bool {
    path.strip_prefix(base)
        .unwrap_or(path)
        .components()
        .any(|component| {
            let name = component.as_os_str();
            name == "node_modules" || name.as_encoded_bytes().starts_with(b".")
        })
}

/// Coalesce an atomic-save burst, returning all paths observed in arrival order.
pub fn coalesce_batch(
    receiver: &Receiver<notify::Result<notify::Event>>,
    first: notify::Result<notify::Event>,
) -> Vec<PathBuf> {
    const QUIET: Duration = Duration::from_millis(2);
    const CAP: Duration = Duration::from_millis(250);
    let mut paths = collect_paths(first);
    let cap_at = Instant::now() + CAP;
    loop {
        let window = QUIET.min(cap_at.saturating_duration_since(Instant::now()));
        if window.is_zero() {
            break;
        }
        match receiver.recv_timeout(window) {
            Ok(event) => paths.extend(collect_paths(event)),
            Err(_) => break,
        }
    }
    paths
}

fn collect_paths(event: notify::Result<notify::Event>) -> Vec<PathBuf> {
    event.map_or_else(|_| Vec::new(), |event| event.paths)
}

pub fn is_module_path(path: &Path) -> bool {
    if path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".diffpack-output" | ".diffpack-next" | "node_modules" | ".git")
        )
    }) || path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
    {
        return false;
    }
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some(
            "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "json" | "css" | "scss" | "sass" | "less"
        )
    )
}

pub fn is_stylesheet_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("css" | "scss" | "sass" | "less" | "styl")
    )
}

pub fn is_config_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    name == "package.json" || name.starts_with("tsconfig") || name.starts_with("diffpack.config")
}

pub fn source_dir(project_root: &Path) -> &'static str {
    if project_root.join("src").is_dir() {
        "src"
    } else {
        "."
    }
}

pub fn display_relative(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_source_config_and_generated_paths() {
        assert!(is_module_path(Path::new("/app/src/view.tsx")));
        assert!(!is_module_path(Path::new("/app/node_modules/pkg/view.js")));
        assert!(is_config_file(Path::new("/app/tsconfig.json")));
        assert!(!is_config_file(Path::new("/app/src/data.json")));
        assert!(is_dependency_or_generated(
            Path::new("/app/.diffpack-output/client.js"),
            Path::new("/app")
        ));
    }

    #[test]
    fn recursive_roots_cover_equal_and_descendant_roots() {
        let recursive = |path: &str| (PathBuf::from(path), RecursiveMode::Recursive);
        let top_level = |path: &str| (PathBuf::from(path), RecursiveMode::NonRecursive);
        let covered = vec![recursive("/app/src"), top_level("/app")];
        assert!(uncovered_roots(&[recursive("/app/src")], &covered).is_empty());
        assert!(uncovered_roots(&[recursive("/app/src/routes")], &covered).is_empty());
        assert_eq!(
            uncovered_roots(&[recursive("/app/packages")], &covered),
            vec![recursive("/app/packages")],
        );
    }
}
