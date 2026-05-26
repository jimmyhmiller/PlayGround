//! Light-touch in-process symbolication.
//!
//! Parses the static symbol table (mach-O `LC_SYMTAB`, ELF `.symtab` /
//! `.dynsym`) of binaries pointed to by `LibMapping` events. No DWARF, no
//! source lines, no inlining info — just function names. That covers most of
//! what a flame graph cares about, fits in ~150 lines, and avoids pulling in
//! `wholesym`/tokio.
//!
//! Loading runs on a background thread so the live stream's reader never
//! blocks on disk I/O. The aggregator's `frame_label` reads the cache under
//! an `RwLock`; misses fall through to the lib+0xRVA fallback.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use crossbeam_channel::{unbounded, Sender};
use object::{Object, ObjectSymbol};

/// One lib's symbols, sorted ascending by RVA. The binary search in
/// `lookup_in_lib` finds the largest entry ≤ the queried RVA — i.e. the
/// containing function. We don't store function ends so we can't reject
/// queries that fall in inter-function padding; that's a tolerable false-
/// attribution for a live flame graph.
#[derive(Default, Clone, Debug)]
pub struct LibSymbols {
    pub entries: Vec<(u32, String)>,
}

pub struct SymbolStore {
    /// Path → loaded symbols. Inserted by the loader thread, read by the
    /// aggregator on each `frame_label` call.
    libs: RwLock<HashMap<PathBuf, LibSymbols>>,
    /// Paths currently queued or being loaded. Prevents double-queueing the
    /// same path when samply emits the same lib for multiple processes.
    pending: Mutex<HashSet<PathBuf>>,
    /// Channel into the loader thread.
    tx: Sender<PathBuf>,
    /// Bumped each time a lib's symbols land. Combined with the
    /// aggregator's event-version so the snapshot builder rebuilds the
    /// Profile when fresh symbols arrive even if no new events did.
    version: AtomicU64,
}

impl SymbolStore {
    pub fn new() -> Arc<Self> {
        let (tx, rx) = unbounded::<PathBuf>();
        let store = Arc::new(Self {
            libs: RwLock::new(HashMap::new()),
            pending: Mutex::new(HashSet::new()),
            tx,
            version: AtomicU64::new(0),
        });
        let s = store.clone();
        thread::Builder::new()
            .name("flame-live-symbols".into())
            .spawn(move || {
                while let Ok(path) = rx.recv() {
                    let result = load_symbols(&path);
                    let symbol_count = result.entries.len();
                    if let Ok(mut libs) = s.libs.write() {
                        libs.insert(path.clone(), result);
                    }
                    if let Ok(mut pending) = s.pending.lock() {
                        pending.remove(&path);
                    }
                    s.version.fetch_add(1, Ordering::Release);
                    log::debug!(
                        "flame-live: loaded {} symbols from {}",
                        symbol_count,
                        path.display()
                    );
                }
            })
            .expect("spawn flame-live-symbols thread");
        store
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Queue `path` for symbol loading if not already loaded or in-flight.
    /// Cheap: only takes the pending-set lock briefly.
    pub fn request(&self, path: &str) {
        let pb = PathBuf::from(path);
        if let Ok(libs) = self.libs.read() {
            if libs.contains_key(&pb) {
                return;
            }
        }
        let mut pending = match self.pending.lock() {
            Ok(p) => p,
            Err(_) => return,
        };
        if pending.contains(&pb) {
            return;
        }
        pending.insert(pb.clone());
        let _ = self.tx.send(pb);
    }

    /// Look up the symbol whose RVA is the largest one ≤ `rva` in `path`.
    /// Returns `None` if the lib isn't loaded yet or has no symbols.
    pub fn lookup(&self, path: &str, rva: u32) -> Option<String> {
        let pb = PathBuf::from(path);
        let libs = self.libs.read().ok()?;
        let lib = libs.get(&pb)?;
        if lib.entries.is_empty() {
            return None;
        }
        let idx = lib.entries.partition_point(|(r, _)| *r <= rva);
        if idx == 0 {
            return None;
        }
        Some(lib.entries[idx - 1].1.clone())
    }
}

fn load_symbols(path: &PathBuf) -> LibSymbols {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            log::debug!(
                "flame-live: symbol load failed: read {}: {e}",
                path.display()
            );
            return LibSymbols::default();
        }
    };
    let file = match object::File::parse(&*data) {
        Ok(f) => f,
        Err(e) => {
            log::debug!(
                "flame-live: symbol load failed: parse {}: {e}",
                path.display()
            );
            return LibSymbols::default();
        }
    };
    let base = file.relative_address_base();
    let mut entries: Vec<(u32, String)> = Vec::new();
    for sym in file.symbols() {
        if !sym.is_definition() {
            continue;
        }
        let raw = match sym.name() {
            Ok(n) if !n.is_empty() => n,
            _ => continue,
        };
        let rva = sym.address().saturating_sub(base) as u32;
        entries.push((rva, demangle(raw)));
    }
    // Dyld's exported-symbol table (also in `dynsym`-equivalent) can have
    // duplicates / aliases at the same RVA. Sort + dedup by RVA, keeping the
    // first name (typically the canonical one).
    entries.sort_by_key(|(r, _)| *r);
    entries.dedup_by_key(|(r, _)| *r);
    LibSymbols { entries }
}

/// Try Rust mangling, then C++ Itanium, then fall back to the raw name with
/// a leading underscore stripped (mach-O convention prepends `_` to every
/// C symbol).
fn demangle(name: &str) -> String {
    if let Ok(d) = rustc_demangle::try_demangle(name) {
        return format!("{d:#}");
    }
    if let Ok(d) = cpp_demangle::Symbol::new(name) {
        return d.to_string();
    }
    if let Some(stripped) = name.strip_prefix('_') {
        return stripped.to_string();
    }
    name.to_string()
}
