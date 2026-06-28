//! In-process HPROF heap dump: snapshot the live set, recover types + layout
//! from DWARF, build the heap reference graph, and write it as a JVM `.hprof`
//! that opens in Eclipse MAT / VisualVM.
//!
//! **Consistency via `fork()`.** A heap dump must be a point-in-time view, but
//! the target's other threads keep mutating memory while we walk it (freeing
//! allocations we're about to read, rewriting pointers). So — like Redis BGSAVE
//! and `gcore` — we `fork()` and build the dump in the **child**, whose memory is
//! a copy-on-write *frozen* image: it can't change underneath us, and the parent
//! (the program) is never paused. The heavy DWARF oracle is built in the parent
//! *before* the fork so the child allocates as little as possible (the child
//! can't safely `malloc` if a parent thread held the allocator lock at fork —
//! the usual fork-in-a-threaded-program caveat).

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

    // Snapshot the heap *first*, so the oracle's own (post-snapshot) allocations
    // don't pollute the captured live set.
    let mut snap = mem::snapshot();
    eprintln!(
        "[memscope] heap dump: {} live allocations ({}) snapshotted in {:.1}s — recovering types…",
        snap.live.len(),
        human(snap.total_live_bytes),
        t0.elapsed().as_secs_f64()
    );

    // Build the DWARF oracle in the PARENT (the heavy allocator), before the
    // fork, so the child's frozen image needs as little allocation as possible.
    let oracle = TypeOracle::for_current_process()
        .map_err(|e| io::Error::other(format!("type resolution unavailable: {e}")))?;
    oracle.resolve_snapshot(&mut snap);
    eprintln!("[memscope] heap dump: types resolved in {:.1}s — forking a frozen snapshot…", t0.elapsed().as_secs_f64());

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

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // A pipe to carry the child's HprofStats back to the parent.
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
            // produce something rather than nothing.
            eprintln!("[memscope] heap dump: fork failed, falling back to in-process dump");
            build_and_write(path, &nodes, oracle.layout(), now_ms)
        }
        0 => {
            // CHILD: our view of memory is now a frozen copy-on-write image.
            // Build the graph + write the hprof against it, report stats up the
            // pipe, and _exit WITHOUT running atexit handlers (which would
            // re-trigger a dump or double-flush the parent's buffers).
            unsafe { libc::close(rd) };
            let code = match build_and_write(path, &nodes, oracle.layout(), now_ms) {
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
            // PARENT: read the child's stats, reap it, and keep running.
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
