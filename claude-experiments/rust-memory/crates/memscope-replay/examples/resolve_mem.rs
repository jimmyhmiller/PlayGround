//! Memory-invariant probe (plan §4.D): resolve the first `N` distinct addresses
//! of a recording with the constant-memory resolver and print peak RSS.
//!
//!   resolve_mem <recording.mscope> <N> [shuffle]
//!
//! Run as a subprocess for several N; peak RSS must be ~flat in N (that's what
//! "constant memory" means). `shuffle` randomizes address order to confirm
//! memory stays bounded even when units get re-touched.

use memscope_replay::read_recording_raw;
use memscope_symbols::resolve_raw_sites_targeted;

fn peak_rss_bytes() -> u64 {
    // macOS: ru_maxrss is bytes; Linux: kilobytes.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0;
        }
        let m = ru.ru_maxrss as u64;
        if cfg!(target_os = "macos") { m } else { m * 1024 }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let file = args.next().expect("usage: resolve_mem <recording> <N> [shuffle]");
    let n: usize = args.next().and_then(|s| s.parse().ok()).expect("N");
    let shuffle = args.next().as_deref() == Some("shuffle");

    let rec = read_recording_raw(&file).expect("read");
    let mut sites: Vec<(u32, Vec<u64>)> =
        rec.raw_sites.iter().map(|(k, v)| (*k, v.clone())).collect();
    sites.sort_by_key(|(k, _)| *k);
    if shuffle {
        // Cheap deterministic shuffle (no rng dep): reverse + stride interleave.
        sites.reverse();
        let mid = sites.len() / 2;
        let mut interleaved = Vec::with_capacity(sites.len());
        for i in 0..mid {
            interleaved.push(sites[i].clone());
            interleaved.push(sites[mid + i].clone());
        }
        sites = interleaved;
    }
    sites.truncate(n);

    let exe = std::path::Path::new(&rec.exe);
    let resolved = resolve_raw_sites_targeted(exe, rec.slide, &sites).expect("resolve");

    // Touch the result so it isn't optimized away.
    let typed = resolved.values().filter(|r| r.element_type.is_some()).count();
    let peak = peak_rss_bytes();
    println!(
        "N={:<8} resolved={:<8} typed={:<8} peak_rss={} MB",
        sites.len(),
        resolved.len(),
        typed,
        peak / (1024 * 1024)
    );
}
