use anyhow::{Context, Result};
use notify::{Watcher, RecursiveMode, Event, EventKind};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use std::time::Duration;

pub struct FileWatcher {
    _watcher: Box<dyn Watcher>,
    receiver: mpsc::Receiver<FileEvent>,
}

#[derive(Debug, Clone)]
pub enum FileEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Removed(PathBuf),
}

impl FileWatcher {
    pub fn new<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let (tx, rx) = mpsc::channel(1000);

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                let file_event = match event.kind {
                    EventKind::Create(_) => {
                        event.paths.into_iter()
                            .map(FileEvent::Created)
                            .collect::<Vec<_>>()
                    }
                    EventKind::Modify(_) => {
                        event.paths.into_iter()
                            .map(FileEvent::Modified)
                            .collect::<Vec<_>>()
                    }
                    EventKind::Remove(_) => {
                        event.paths.into_iter()
                            .map(FileEvent::Removed)
                            .collect::<Vec<_>>()
                    }
                    _ => Vec::new(),
                };

                for evt in file_event {
                    let _ = tx.blocking_send(evt);
                }
            }
        })
        .context("Failed to create file watcher")?;

        // Watch all specified paths
        for path in paths {
            watcher.watch(path.as_ref(), RecursiveMode::Recursive)
                .context(format!("Failed to watch path: {}", path.as_ref().display()))?;
            log::info!("Watching directory: {}", path.as_ref().display());
        }

        Ok(Self {
            _watcher: Box::new(watcher),
            receiver: rx,
        })
    }

    pub async fn next_event(&mut self) -> Option<FileEvent> {
        self.receiver.recv().await
    }

    pub fn should_ignore(&self, path: &Path, ignore_extensions: &[String]) -> bool {
        // Ignore hidden files and directories
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') {
                return true;
            }
        }

        // Ignore by extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext_with_dot = format!(".{}", ext);
            if ignore_extensions.contains(&ext_with_dot) {
                return true;
            }
        }

        // Ignore directories
        if path.is_dir() {
            return true;
        }

        false
    }
}

/// Debouncer to handle rapid file changes
pub struct Debouncer {
    last_event_time: std::collections::HashMap<PathBuf, std::time::Instant>,
    debounce_duration: Duration,
}

impl Debouncer {
    pub fn new(debounce_secs: u64) -> Self {
        Self {
            last_event_time: std::collections::HashMap::new(),
            debounce_duration: Duration::from_secs(debounce_secs),
        }
    }

    pub fn should_process(&mut self, path: &Path) -> bool {
        let now = std::time::Instant::now();
        let path_buf = path.to_path_buf();

        if let Some(&last_time) = self.last_event_time.get(&path_buf) {
            if now.duration_since(last_time) < self.debounce_duration {
                // Too soon, skip this event
                return false;
            }
        }

        // Update the last event time
        self.last_event_time.insert(path_buf, now);
        true
    }

    pub fn cleanup_old_entries(&mut self) {
        let now = std::time::Instant::now();
        let debounce_duration = self.debounce_duration;

        self.last_event_time.retain(|_, &mut last_time| {
            now.duration_since(last_time) < debounce_duration * 10
        });
    }
}
