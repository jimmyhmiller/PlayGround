//! In-process HPROF heap dump: snapshot the live set, recover types + layout
//! from DWARF, build the heap reference graph, and write it as a JVM `.hprof`
//! that opens in Eclipse MAT / VisualVM.
//!
//! **Consistency via `fork()`.** A heap dump must be a point-in-time view, but
//! the target's other threads keep mutating memory while we walk it (freeing
//! allocations we're about to read, rewriting pointers). So — like Redis BGSAVE
//! and `gcore` — we `fork()` and build the dump in the **child**, whose memory is
//! a copy-on-write *frozen* image: it can't change underneath us, and the parent
//! (the program) is never paused.
//!
//! **Everything expensive runs in the child.** We fork *immediately* after the
//! (fast) live-set snapshot, then build the DWARF oracle, resolve types, and
//! write the hprof entirely in the child. The reason is survival: when the Rust
//! code lives in a dynamically-loaded module (a node `.node` addon, a Python
//! extension) the host may `process.exit()` the instant its work finishes —
//! killing the parent — while parsing a debug=2 DWARF (100+ MB) and resolving
//! millions of sites can take longer than that. Building types in the parent
//! *before* the fork (the old ordering) meant the host tore us down mid-resolve
//! and no hprof was ever written. The forked child is an independent process on a
//! frozen image: it survives the host exiting and can take minutes if it needs
//! to. We `setsid()` it so a Ctrl-C / process-group signal to the host doesn't
//! take it down with it.
//!
//! The tradeoff is that the child now does heavy allocation post-`fork()` — the
//! classic fork-in-a-threaded-program hazard, where a parent thread holding the
//! allocator lock at fork time would deadlock the child's first `malloc`. In
//! practice the target here (turbopack/TurboMalloc) is mimalloc, whose per-thread
//! heaps take no global lock on the fast path, so this is far safer than with
//! glibc malloc — and `build_and_write` already allocated in the child, so the
//! pattern was already in play; we've only made it do more.

use std::collections::HashMap;
use std::io::{self, BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use memscope_core as mem;
use memscope_graph::NodeInput;
use memscope_proto::SiteInfo;
use memscope_symbols::{LayoutIndex, MemReader, TypeOracle};

pub use memscope_hprof::HprofStats;

extern "C" {
    /// Copy `size` bytes from `address` in `task` into `data`. Returns a Mach
    /// error (non-zero) — *not* a fault — when `address` is unmapped, which is
    /// exactly what we need when reading a heap whose live set may name a
    /// since-freed address.
    fn mach_vm_read_overwrite(
        task: libc::mach_port_t,
        address: u64,
        size: u64,
        data: u64,
        out_size: *mut u64,
    ) -> libc::kern_return_t;
}

/// A [`MemReader`] that reads the current process's memory **safely** via Mach,
/// returning nothing for unmapped addresses instead of segfaulting.
struct SafeReader {
    task: libc::mach_port_t,
}

impl SafeReader {
    fn new() -> Self {
        #[allow(deprecated)]
        SafeReader { task: unsafe { libc::mach_task_self() } }
    }
}

impl MemReader for SafeReader {
    fn read_uint(&self, addr: u64, size: u64) -> Option<u64> {
        let mut buf = [0u8; 8];
        let mut out = 0u64;
        let kr = unsafe {
            mach_vm_read_overwrite(self.task, addr, size, buf.as_mut_ptr() as u64, &mut out)
        };
        if kr != 0 || out < size {
            return None;
        }
        Some(match size {
            1 => buf[0] as u64,
            2 => u16::from_le_bytes([buf[0], buf[1]]) as u64,
            4 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as u64,
            _ => u64::from_le_bytes(buf),
        })
    }

    fn read_into(&self, addr: u64, buf: &mut [u8]) -> usize {
        if buf.is_empty() {
            return 0;
        }
        let mut out = 0u64;
        let kr = unsafe {
            mach_vm_read_overwrite(self.task, addr, buf.len() as u64, buf.as_mut_ptr() as u64, &mut out)
        };
        if kr != 0 {
            0
        } else {
            out as usize
        }
    }
}

/// Write a type-resolved HPROF heap dump of the current process to `path`.
pub fn heap_dump(path: &str) -> io::Result<HprofStats> {
    let t0 = std::time::Instant::now();
    // Freeze the event stream for the duration so the snapshot is stable and the
    // target's threads don't stall on ring backpressure while we walk the heap.
    let prev_mode = mem::stats().mode;
    mem::set_mode(mem::Mode::Off);
    let _restore = ModeRestore(prev_mode);

    // Snapshot the heap *first*, so the (post-snapshot) allocations of type
    // recovery don't pollute the captured live set. This is the only expensive
    // step we do before forking — it's fast (~0.1s for millions of allocations).
    let snap = mem::snapshot();
    eprintln!(
        "[memscope] heap dump: {} live allocations ({}) snapshotted in {:.1}s — forking before type recovery…",
        snap.live.len(),
        human(snap.total_live_bytes),
        t0.elapsed().as_secs_f64()
    );

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // A pipe to carry the child's HprofStats back to the parent — best-effort:
    // if the host tears the parent down before the child finishes, the orphaned
    // child still writes the hprof and logs its own completion.
    let mut fds = [0 as libc::c_int; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let (rd, wr) = (fds[0], fds[1]);

    match unsafe { libc::fork() } {
        -1 => {
            unsafe {
                libc::close(rd);
                libc::close(wr);
            }
            // Fork failed — fall back to an in-process (racy) dump so we still
            // produce something rather than nothing. Type recovery runs inline.
            eprintln!("[memscope] heap dump: fork failed, falling back to in-process dump");
            resolve_and_write(path, snap, now_ms, t0)
        }
        0 => {
            // CHILD: our view of memory is now a frozen copy-on-write image.
            // Detach into our own session so a Ctrl-C / process-group signal to
            // the host — or the host simply exiting — doesn't take us down while
            // we're still parsing DWARF and writing the dump (which can outlast
            // the host by minutes). Build the oracle, resolve types, build the
            // graph, write the hprof, report stats up the pipe, and _exit WITHOUT
            // running atexit handlers (which would re-trigger a dump or
            // double-flush the parent's buffers).
            unsafe {
                libc::close(rd);
                libc::setsid();
            }
            let code = match resolve_and_write(path, snap, now_ms, t0) {
                Ok(s) => {
                    let mut buf = [0u8; 32];
                    buf[0..8].copy_from_slice(&s.objects.to_le_bytes());
                    buf[8..16].copy_from_slice(&s.classes.to_le_bytes());
                    buf[16..24].copy_from_slice(&s.roots.to_le_bytes());
                    buf[24..32].copy_from_slice(&s.bytes_written.to_le_bytes());
                    unsafe { libc::write(wr, buf.as_ptr() as *const libc::c_void, 32) };
                    0
                }
                Err(e) => {
                    eprintln!("[memscope] heap dump child failed: {e}");
                    1
                }
            };
            unsafe { libc::_exit(code) };
        }
        pid => {
            // PARENT: try to read the child's stats and reap it, then keep
            // running. This blocks until the child finishes *or* the host tears
            // us down — either is fine: the child writes the file regardless.
            unsafe { libc::close(wr) };
            let stats = read_stats(rd);
            unsafe { libc::close(rd) };
            let mut status = 0;
            unsafe { libc::waitpid(pid, &mut status, 0) };
            match stats {
                Some(s) => {
                    eprintln!(
                        "[memscope] wrote heap dump {path}: {} objects, {} classes, {} roots, {} in {:.1}s",
                        s.objects,
                        s.classes,
                        s.roots,
                        human(s.bytes_written),
                        t0.elapsed().as_secs_f64()
                    );
                    Ok(s)
                }
                None => Err(io::Error::other("heap dump child produced no output")),
            }
        }
    }
}

/// Recover types from DWARF, project the snapshot into graph nodes, and write the
/// hprof. This is the **expensive** half of a dump — parsing a large debug=2
/// DWARF and resolving millions of sites — so it runs in the forked child (see
/// the module docs), or in-process on fork failure.
fn resolve_and_write(
    path: &str,
    mut snap: mem::Snapshot,
    now_ms: u64,
    t0: std::time::Instant,
) -> io::Result<HprofStats> {
    // Build the DWARF oracle. When the Rust code lives in a dynamically-loaded
    // module (a node/.node addon, a Python extension, etc.) rather than the host
    // exe, point at it via MEMSCOPE_BINARY so we resolve the addon's DWARF
    // instead of the host's (`current_exe()` would be `node`, which has no Rust
    // DWARF and makes dsymutil fail).
    eprintln!(
        "[memscope] heap dump: recovering types (pid {})…",
        std::process::id()
    );
    let oracle = match std::env::var("MEMSCOPE_BINARY") {
        Ok(p) if !p.is_empty() => TypeOracle::from_binary(std::path::Path::new(&p)),
        _ => TypeOracle::for_current_process(),
    }
    .map_err(|e| io::Error::other(format!("type resolution unavailable: {e}")))?;
    oracle.resolve_snapshot(&mut snap);
    eprintln!(
        "[memscope] heap dump: types resolved in {:.1}s — building graph…",
        t0.elapsed().as_secs_f64()
    );

    let type_name: HashMap<u32, String> =
        snap.types.iter().map(|t| (t.id, t.name.clone())).collect();
    let site: HashMap<u32, &SiteInfo> = snap.sites.iter().map(|s| (s.id, s)).collect();
    let nodes: Vec<NodeInput> = snap
        .live
        .iter()
        .map(|l| {
            let (ty, shape) = site
                .get(&l.site.0)
                .map(|s| (type_name.get(&s.ty.0).cloned(), s.shape))
                .unwrap_or((None, None));
            NodeInput { addr: l.addr, size: l.size, type_name: ty, shape }
        })
        .collect();

    // MEMSCOPE_TYPE_BREAKDOWN=1: print the authoritative live-bytes-by-type using the REAL
    // recorded allocation sizes (n.size). Unlike the hprof — which writes typed objects at their
    // DWARF layout size and only untyped allocations at true size — this sums every allocation's
    // actual bytes, so the total matches the tracked live set. Untyped raw buffers bucket as
    // "<untyped>".
    if std::env::var("MEMSCOPE_TYPE_BREAKDOWN").is_ok_and(|v| !v.is_empty() && v != "0") {
        let mut by_type: HashMap<&str, (u64, u64)> = HashMap::new();
        let mut total: u64 = 0;
        for n in &nodes {
            let name = n.type_name.as_deref().unwrap_or("<untyped raw buffer>");
            let e = by_type.entry(name).or_insert((0, 0));
            e.0 += 1;
            e.1 += n.size;
            total += n.size;
        }
        let mut v: Vec<(&str, u64, u64)> =
            by_type.iter().map(|(k, (c, b))| (*k, *c, *b)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        eprintln!(
            "[memscope] === live-bytes-by-type (real n.size) — total {:.1} MB across {} allocs ===",
            total as f64 / 1_048_576.0,
            nodes.len()
        );
        for (name, count, bytes) in v.iter().take(35) {
            eprintln!(
                "[memscope]  {:>9.1} MB  {:>9}  {}",
                *bytes as f64 / 1_048_576.0,
                count,
                name
            );
        }
    }

    build_and_write(path, &nodes, oracle.layout(), now_ms)
}

/// Build the reference graph + write the hprof. Runs in the forked child (or, on
/// fork failure, in-process).
fn build_and_write(
    path: &str,
    nodes: &[NodeInput],
    layout: &LayoutIndex,
    now_ms: u64,
) -> io::Result<HprofStats> {
    // Safe reads: the live set may name a since-freed (now unmapped) address, so
    // dereferencing raw would segfault. SafeReader returns nothing instead.
    let reader = SafeReader::new();
    let graph = memscope_graph::build(nodes, layout, &reader);
    eprintln!(
        "[memscope] heap dump: graph built ({} nodes, {} edges) — writing hprof…",
        graph.nodes.len(),
        graph.edges.len()
    );
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    let stats = memscope_hprof::write_hprof(&mut w, &graph, layout, &reader, now_ms)?;
    w.flush()?;
    Ok(stats)
}

/// Read 4 little-endian u64s (objects, classes, roots, bytes) from the child.
fn read_stats(fd: libc::c_int) -> Option<HprofStats> {
    let mut buf = [0u8; 32];
    let mut got = 0usize;
    while got < 32 {
        let n = unsafe {
            libc::read(fd, buf[got..].as_mut_ptr() as *mut libc::c_void, 32 - got)
        };
        if n <= 0 {
            return None;
        }
        got += n as usize;
    }
    Some(HprofStats {
        objects: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        classes: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        roots: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
        bytes_written: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
    })
}

/// Restores the recorder mode when the dump finishes (or errors out).
struct ModeRestore(mem::Mode);
impl Drop for ModeRestore {
    fn drop(&mut self) {
        mem::set_mode(self.0);
    }
}

fn human(n: u64) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}
