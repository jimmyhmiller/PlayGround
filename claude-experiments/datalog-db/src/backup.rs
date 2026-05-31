//! Periodic checkpoint backups for `Database`.
//!
//! A `BackupScheduler` runs in a background thread, calling
//! `Database::create_checkpoint` at a configured interval. Each checkpoint
//! lands in `<root>/<unix_millis>/` (a sibling of the live data-dir,
//! same filesystem — checkpoints are hard-link based). Old checkpoints
//! beyond `retain` are pruned oldest-first after each new one is taken.
//!
//! Checkpoints are crash-consistent points-in-time. To restore: stop the
//! server, point `--data-dir` at the checkpoint directory, restart.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tracing::{info, warn};

use crate::db::Database;

#[derive(Debug, thiserror::Error)]
pub enum BackupError {
    #[error("backup IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database error: {0}")]
    Db(#[from] crate::db::DbError),
    #[error("backup root must not be inside data-dir")]
    RootInsideDataDir,
}

pub type Result<T> = std::result::Result<T, BackupError>;

/// Create one checkpoint under `root`. The new directory is named with
/// the current unix-millis timestamp; if that name already exists (e.g.
/// two backups within the same millisecond), a `_N` suffix is appended.
/// Returns the absolute path of the created checkpoint.
pub fn create_checkpoint(db: &Database, root: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(root)?;

    let base = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let mut path = root.join(format!("{:013}", base));
    let mut suffix = 1u32;
    while path.exists() {
        path = root.join(format!("{:013}_{}", base, suffix));
        suffix += 1;
        if suffix > 1000 {
            return Err(BackupError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not find a unique checkpoint name",
            )));
        }
    }

    db.create_checkpoint(&path)?;
    Ok(path)
}

/// List all checkpoint directories under `root`, oldest-first. A directory
/// counts as a checkpoint if its name starts with 13 digits (a millis
/// timestamp). Non-matching entries are ignored so the root can also
/// contain unrelated files without confusing pruning.
pub fn list_checkpoints(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.len() >= 13 && s.as_bytes()[..13].iter().all(|b| b.is_ascii_digit()))
                    .unwrap_or(false)
        })
        .collect();
    // Sort oldest-first by (base millis, suffix) numerically. A plain string
    // sort would order `..._10` before `..._2` for checkpoints that collide
    // within the same millisecond.
    entries.sort_by_key(|p| checkpoint_sort_key(p));
    Ok(entries)
}

/// Parse a checkpoint directory name (`{:013}` optionally `_{suffix}`) into a
/// numeric `(base_millis, suffix)` sort key. Unparsable names sort first.
fn checkpoint_sort_key(p: &Path) -> (u64, u32) {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let base = name.get(..13).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    // Characters after the 13-digit base and a single `_` separator.
    let suffix = name.get(14..).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    (base, suffix)
}

/// Delete the oldest checkpoints in `root` until at most `retain` remain.
/// Returns the number of checkpoints removed. `retain == 0` deletes them
/// all.
pub fn prune_checkpoints(root: &Path, retain: usize) -> Result<usize> {
    let mut entries = list_checkpoints(root)?;
    if entries.len() <= retain {
        return Ok(0);
    }
    let to_remove = entries.len() - retain;
    let mut removed = 0usize;
    for path in entries.drain(..to_remove) {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => removed += 1,
            Err(e) => warn!("failed to remove old checkpoint {:?}: {}", path, e),
        }
    }
    Ok(removed)
}

/// Configuration for the periodic backup scheduler.
#[derive(Debug, Clone)]
pub struct BackupSchedulerConfig {
    /// Directory under which timestamped checkpoint folders are written.
    /// Must be on the same filesystem as the live data-dir (checkpoints
    /// are hard-linked, not copied).
    pub root: PathBuf,
    /// Time between checkpoints.
    pub interval: Duration,
    /// How many checkpoints to keep. Older ones are pruned after each
    /// successful new backup.
    pub retain: usize,
}

/// Handle for a running scheduler. Drop or call `stop()` to terminate
/// the background thread cleanly.
pub struct BackupSchedulerHandle {
    stop_flag: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl BackupSchedulerHandle {
    /// Signal the scheduler to stop and join its thread.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for BackupSchedulerHandle {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Spawn a background thread that takes a checkpoint every `config.interval`
/// and prunes the oldest down to `config.retain`. Failures are logged but
/// do not stop the schedule — a transient disk-full or rename collision
/// won't kill the loop.
pub fn spawn_backup_scheduler(
    db: Arc<Database>,
    config: BackupSchedulerConfig,
) -> BackupSchedulerHandle {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    let thread = std::thread::Builder::new()
        .name("backup-scheduler".into())
        .spawn(move || {
            info!(
                "Backup scheduler started: root={:?} interval={:?} retain={}",
                config.root, config.interval, config.retain
            );
            // Tick using a deadline so a long checkpoint doesn't drift the
            // schedule. If a checkpoint takes longer than the interval,
            // the next one fires immediately.
            let mut next_at = Instant::now() + config.interval;
            while !stop_flag_clone.load(Ordering::SeqCst) {
                // Wake up to check the stop flag at least once a second so
                // shutdown is responsive even with long intervals.
                let now = Instant::now();
                if now < next_at {
                    let sleep_for = (next_at - now).min(Duration::from_secs(1));
                    std::thread::sleep(sleep_for);
                    continue;
                }

                match create_checkpoint(&db, &config.root) {
                    Ok(path) => {
                        info!("Backup checkpoint created at {:?}", path);
                        match prune_checkpoints(&config.root, config.retain) {
                            Ok(0) => {}
                            Ok(n) => info!("Pruned {} old backup checkpoint(s)", n),
                            Err(e) => warn!("Backup prune failed: {}", e),
                        }
                    }
                    Err(e) => warn!("Backup checkpoint failed: {}", e),
                }

                next_at = Instant::now() + config.interval;
            }
            info!("Backup scheduler stopped");
        })
        .expect("spawn backup-scheduler thread");

    BackupSchedulerHandle {
        stop_flag,
        thread: Some(thread),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_checkpoints_orders_numerically() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Same-millisecond checkpoints with numeric suffixes, plus a later one.
        for name in [
            "0000000001000",
            "0000000001000_2",
            "0000000001000_10",
            "0000000002000",
            "not-a-checkpoint", // ignored
        ] {
            std::fs::create_dir(root.join(name)).unwrap();
        }
        let listed: Vec<String> = list_checkpoints(root)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        // Numeric order: suffix 2 before 10 (a string sort would invert these).
        assert_eq!(
            listed,
            vec![
                "0000000001000".to_string(),
                "0000000001000_2".to_string(),
                "0000000001000_10".to_string(),
                "0000000002000".to_string(),
            ]
        );
    }
}
