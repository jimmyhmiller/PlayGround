//! Allocation-stream detectors: turn a [`Recording`] into ranked, source-located
//! **findings** an AI (or a human) can act on, rather than a picture to eyeball.
//!
//! Everything here is a pure pass over the event stream + the reconstructed final
//! live set — no live process, no DWARF, no graph. Each detector states an
//! explicit fingerprint and a **fix class** from a closed vocabulary so a
//! downstream agent can branch on it deterministically.

use std::collections::HashMap;

use memscope_proto::EventKind;

use crate::Recording;

/// Per-`(type, site)` aggregate stats, computed in one pass over the stream.
#[derive(Clone, Debug)]
pub struct SiteStats {
    pub site: u32,
    /// Resolved type label (e.g. `Boxed<Particle>`).
    pub label: String,
    /// Allocations charged to this site (`Alloc` + `ReallocGrow` events).
    pub alloc_count: u64,
    /// Total bytes allocated at this site over the whole run.
    pub alloc_bytes: u64,
    /// How many of those were reallocation-grows (Vec/String growth copies).
    pub realloc_count: u64,
    /// Frees attributed to this site (by addr; Dealloc carries no site itself).
    pub free_count: u64,
    /// Sum of attributed-free lifetimes, milliseconds.
    pub lifetime_sum_ms: u64,
    /// A bounded sample of freed-allocation lifetimes (ms) for a median estimate.
    pub lifetime_sample: Vec<u64>,
    /// Allocations from this site still live at end of stream.
    pub live_count: u64,
    pub live_bytes: u64,
}

impl SiteStats {
    fn new(site: u32, label: String) -> Self {
        SiteStats {
            site,
            label,
            alloc_count: 0,
            alloc_bytes: 0,
            realloc_count: 0,
            free_count: 0,
            lifetime_sum_ms: 0,
            lifetime_sample: Vec::new(),
            live_count: 0,
            live_bytes: 0,
        }
    }

    /// Median freed-allocation lifetime (ms), from the bounded sample. `None`
    /// when nothing from this site was observed to free.
    pub fn median_lifetime_ms(&self) -> Option<u64> {
        if self.lifetime_sample.is_empty() {
            return None;
        }
        let mut v = self.lifetime_sample.clone();
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    /// Fraction of this site's allocations that were freed (0..1).
    pub fn freed_fraction(&self) -> f64 {
        if self.alloc_count == 0 {
            return 0.0;
        }
        self.free_count as f64 / self.alloc_count as f64
    }

    /// True when the recovered type label is a `Box<T>`.
    pub fn is_boxed(&self) -> bool {
        self.label.starts_with("Boxed<")
    }
}

const LIFETIME_SAMPLE_CAP: usize = 1024;

/// One pass over the stream: aggregate per-site allocation/free/realloc stats and
/// reconstruct the final live set. Frees (which carry no site) are attributed by
/// tracking `addr -> (site, size, alloc-ts)` as allocations happen.
pub fn site_stats(rec: &Recording) -> Vec<SiteStats> {
    struct Live {
        site: u32,
        size: u64,
        ts: u64,
    }
    let mut live: HashMap<u64, Live> = HashMap::new();
    let mut stats: HashMap<u32, SiteStats> = HashMap::new();

    for e in &rec.events {
        match e.kind {
            EventKind::Alloc | EventKind::ReallocGrow => {
                let s = stats
                    .entry(e.site)
                    .or_insert_with(|| SiteStats::new(e.site, rec.site_label(e.site).to_string()));
                s.alloc_count += 1;
                s.alloc_bytes += e.size;
                if e.kind == EventKind::ReallocGrow {
                    s.realloc_count += 1;
                }
                live.insert(e.addr, Live { site: e.site, size: e.size, ts: e.ts_nanos });
            }
            EventKind::Dealloc => {
                if let Some(l) = live.remove(&e.addr) {
                    let s = stats
                        .entry(l.site)
                        .or_insert_with(|| SiteStats::new(l.site, rec.site_label(l.site).to_string()));
                    s.free_count += 1;
                    let lt = e.ts_nanos.saturating_sub(l.ts) / 1_000_000;
                    s.lifetime_sum_ms += lt;
                    if s.lifetime_sample.len() < LIFETIME_SAMPLE_CAP {
                        s.lifetime_sample.push(lt);
                    }
                }
            }
            EventKind::MetaEnter | EventKind::MetaExit | EventKind::Mark => {}
        }
    }
    // Whatever remains live at end of stream.
    for l in live.values() {
        let s = stats
            .entry(l.site)
            .or_insert_with(|| SiteStats::new(l.site, rec.site_label(l.site).to_string()));
        s.live_count += 1;
        s.live_bytes += l.size;
    }
    stats.into_values().collect()
}

/// A ranked, source-located finding. `severity` (0..1) drives ranking;
/// `confidence` (0..1) is reported separately so a high-impact-but-uncertain
/// finding can be weighed against a small-but-certain one.
#[derive(Clone, Debug)]
pub struct Finding {
    pub detector: &'static str,
    pub severity: f64,
    pub confidence: f64,
    pub title: String,
    pub site: u32,
    /// Resolved type label.
    pub ty: String,
    /// Application boundary location (`func (file:line)`), or empty if unknown.
    pub location: String,
    /// Closed-vocabulary fix class (see module docs).
    pub fix_class: &'static str,
    pub suggestion: String,
    /// Ordered evidence key/values, surfaced verbatim in text + JSON.
    pub evidence: Vec<(&'static str, String)>,
}

// --- detector thresholds (documented so the honesty rules are visible) -------

/// A site retaining at least this many live bytes is leak-eligible.
const LEAK_MIN_LIVE_BYTES: u64 = 64 * 1024;
/// …and freeing at most this fraction of what it allocated.
const LEAK_MAX_FREED_FRACTION: f64 = 0.5;
/// Churn: a site must allocate at least this much total to be worth flagging.
const CHURN_MIN_ALLOC_BYTES: u64 = 256 * 1024;
/// …and its allocated:live ratio must be at least this (mostly transient).
const CHURN_MIN_RATIO: f64 = 8.0;
/// Realloc-thrash: this many grow-copies (well above normal doubling, which is
/// ~log2(n) reallocations for n pushes).
const REALLOC_THRASH_MIN: u64 = 32;
/// Short-lived box: at least this many freed, with a tiny median lifetime.
const SHORTLIVED_MIN_FREES: u64 = 1000;
const SHORTLIVED_MAX_MEDIAN_MS: u64 = 2;

/// Merge `other`'s counts into `s` (same boundary location + type).
fn merge_into(s: &mut SiteStats, other: &SiteStats) {
    s.alloc_count += other.alloc_count;
    s.alloc_bytes += other.alloc_bytes;
    s.realloc_count += other.realloc_count;
    s.free_count += other.free_count;
    s.lifetime_sum_ms += other.lifetime_sum_ms;
    for &lt in &other.lifetime_sample {
        if s.lifetime_sample.len() < LIFETIME_SAMPLE_CAP {
            s.lifetime_sample.push(lt);
        }
    }
    s.live_count += other.live_count;
    s.live_bytes += other.live_bytes;
    s.site = s.site.min(other.site);
}

/// Run all detectors over a recording, returning findings ranked by severity.
///
/// Per-site stats are first **merged by application boundary location + type**
/// (so loop-unrolled call sites that share a source line collapse into one
/// finding) and sites whose allocation originates *inside* the profiler/DWARF
/// machinery are dropped (they're measurement overhead, not the program).
pub fn analyze(rec: &Recording) -> Vec<Finding> {
    let stats = site_stats(rec);

    // key: (boundary location, type label) -> (merged stats, location string)
    let mut merged: HashMap<(String, String), (SiteStats, String)> = HashMap::new();
    for s in stats {
        let frames = rec.sites.get(&s.site).map(|i| i.frames.as_slice()).unwrap_or(&[]);
        if crate::is_profiler_origin(frames) {
            continue;
        }
        let loc = crate::boundary_frame(frames).map(crate::frame_location).unwrap_or_default();
        let key = (loc.clone(), s.label.clone());
        match merged.get_mut(&key) {
            Some((acc, _)) => merge_into(acc, &s),
            None => {
                merged.insert(key, (s, loc));
            }
        }
    }

    let groups: Vec<(SiteStats, String)> = merged.into_values().collect();
    let total_alloc_bytes: u64 = groups.iter().map(|(s, _)| s.alloc_bytes).sum::<u64>().max(1);
    let total_live_bytes: u64 = groups.iter().map(|(s, _)| s.live_bytes).sum::<u64>().max(1);
    let total_frees: u64 = groups.iter().map(|(s, _)| s.free_count).sum::<u64>().max(1);

    let mut out = Vec::new();
    for (s, loc) in &groups {
        detect_leak(s, loc, total_live_bytes, &mut out);
        detect_churn(s, loc, total_alloc_bytes, &mut out);
        detect_realloc_thrash(s, loc, &mut out);
        detect_short_lived_box(s, loc, total_frees, &mut out);
    }
    out.sort_by(|a, b| b.severity.total_cmp(&a.severity));
    out
}

fn detect_leak(s: &SiteStats, loc: &str, total_live_bytes: u64, out: &mut Vec<Finding>) {
    if s.live_bytes < LEAK_MIN_LIVE_BYTES || s.freed_fraction() > LEAK_MAX_FREED_FRACTION {
        return;
    }
    let never_freed = s.free_count == 0;
    let (fix_class, suggestion) = if never_freed {
        ("leak", "Allocated here and never freed; ensure these are dropped, or bound their lifetime.".to_string())
    } else {
        (
            "unbounded-cache",
            "Grows faster than it frees; bound it (LRU/TTL) or drain entries when done.".to_string(),
        )
    };
    out.push(Finding {
        detector: "monotonic-growth",
        severity: s.live_bytes as f64 / total_live_bytes as f64,
        confidence: if never_freed { 0.85 } else { 0.6 },
        title: format!(
            "{}: {} live in {} allocations, {} freed{}",
            s.label,
            human(s.live_bytes),
            s.live_count,
            s.free_count,
            if never_freed { " (never freed)" } else { "" }
        ),
        site: s.site,
        ty: s.label.clone(),
        location: loc.to_string(),
        fix_class,
        suggestion,
        evidence: vec![
            ("live_bytes", s.live_bytes.to_string()),
            ("live_count", s.live_count.to_string()),
            ("allocated", s.alloc_count.to_string()),
            ("freed", s.free_count.to_string()),
        ],
    });
}

fn detect_churn(s: &SiteStats, loc: &str, total_alloc_bytes: u64, out: &mut Vec<Finding>) {
    if s.alloc_bytes < CHURN_MIN_ALLOC_BYTES {
        return;
    }
    let ratio = s.alloc_bytes as f64 / s.live_bytes.max(1) as f64;
    if ratio < CHURN_MIN_RATIO {
        return;
    }
    // When nothing is live the ratio is unbounded — describe it as fully
    // transient rather than printing a meaningless "×<alloc_bytes>".
    let churn_desc = if s.live_bytes == 0 {
        "all freed".to_string()
    } else {
        format!("churn ×{ratio:.0}")
    };
    out.push(Finding {
        detector: "churn-storm",
        severity: s.alloc_bytes as f64 / total_alloc_bytes as f64,
        confidence: 0.8,
        title: format!(
            "{}: {} allocated across {} allocations, {} live ({})",
            s.label,
            human(s.alloc_bytes),
            s.alloc_count,
            human(s.live_bytes),
            churn_desc
        ),
        site: s.site,
        ty: s.label.clone(),
        location: loc.to_string(),
        fix_class: "reuse-buffer",
        suggestion: "Mostly transient — hoist the allocation out of the hot loop, reuse one buffer, or use an arena.".to_string(),
        evidence: vec![
            ("alloc_bytes", s.alloc_bytes.to_string()),
            ("alloc_count", s.alloc_count.to_string()),
            ("live_bytes", s.live_bytes.to_string()),
            ("churn_ratio", if s.live_bytes == 0 { "inf".to_string() } else { format!("{ratio:.1}") }),
        ],
    });
}

fn detect_realloc_thrash(s: &SiteStats, loc: &str, out: &mut Vec<Finding>) {
    if s.realloc_count < REALLOC_THRASH_MIN {
        return;
    }
    out.push(Finding {
        detector: "realloc-thrash",
        severity: (s.realloc_count as f64 / 256.0).min(1.0),
        confidence: 0.7,
        title: format!(
            "{}: {} reallocation-grows (growing a buffer incrementally)",
            s.label, s.realloc_count
        ),
        site: s.site,
        ty: s.label.clone(),
        location: loc.to_string(),
        fix_class: "with_capacity",
        suggestion: "Reserve the final capacity up front (`with_capacity`/`reserve`) to avoid repeated grow-and-copy.".to_string(),
        evidence: vec![
            ("realloc_count", s.realloc_count.to_string()),
            ("alloc_count", s.alloc_count.to_string()),
        ],
    });
}

fn detect_short_lived_box(s: &SiteStats, loc: &str, total_frees: u64, out: &mut Vec<Finding>) {
    // Gate on Box<T> so this stays distinct from churn-storm (which catches the
    // Vec/buffer case): a high-volume, very-short-lived Box is the classic
    // "could live on the stack / in a pool" candidate.
    if !s.is_boxed() || s.free_count < SHORTLIVED_MIN_FREES {
        return;
    }
    let median = match s.median_lifetime_ms() {
        Some(m) if m <= SHORTLIVED_MAX_MEDIAN_MS => m,
        _ => return,
    };
    out.push(Finding {
        detector: "short-lived-box",
        severity: (s.free_count as f64 / total_frees as f64).min(1.0),
        confidence: 0.6,
        title: format!(
            "{}: {} short-lived boxes (median lifetime {} ms)",
            s.label, s.free_count, median
        ),
        site: s.site,
        ty: s.label.clone(),
        location: loc.to_string(),
        fix_class: "box-to-stack",
        suggestion: "Frequently allocated and freed almost immediately — keep it on the stack, or pool/arena the allocations.".to_string(),
        evidence: vec![
            ("freed", s.free_count.to_string()),
            ("median_lifetime_ms", median.to_string()),
        ],
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameMeta, RecEvent, SiteInfo};

    fn ev(kind: EventKind, addr: u64, size: u64, ts: u64, site: u32) -> RecEvent {
        RecEvent { kind, addr, size, ts_nanos: ts, site, thread: 1 }
    }

    fn rec_with(sites: &[(u32, &str)], events: Vec<RecEvent>) -> Recording {
        let mut s = HashMap::new();
        for (id, label) in sites {
            s.insert(*id, SiteInfo { label: (*label).to_string(), frames: vec![] });
        }
        Recording { sites: s, meta: HashMap::new(), marks: HashMap::new(), events }
    }

    #[test]
    fn leak_detector_flags_never_freed() {
        // 100 allocations of 1 KiB at site 1, none freed.
        let mut events = Vec::new();
        for i in 0..100u64 {
            events.push(ev(EventKind::Alloc, 0x1000 + i, 1024, i, 1));
        }
        let rec = rec_with(&[(1, "Boxed<Session>")], events);
        let f = analyze(&rec);
        let leak = f.iter().find(|f| f.detector == "monotonic-growth").unwrap();
        assert_eq!(leak.fix_class, "leak");
        assert_eq!(leak.site, 1);
    }

    #[test]
    fn churn_detector_flags_transient_storm() {
        // 10k allocations of 1 KiB, each freed immediately -> ~0 live.
        let mut events = Vec::new();
        for i in 0..10_000u64 {
            events.push(ev(EventKind::Alloc, 0x10, 1024, i * 2, 2));
            events.push(ev(EventKind::Dealloc, 0x10, 1024, i * 2 + 1, 0));
        }
        let rec = rec_with(&[(2, "Vec<u8>")], events);
        let f = analyze(&rec);
        let churn = f.iter().find(|f| f.detector == "churn-storm").unwrap();
        assert_eq!(churn.fix_class, "reuse-buffer");
        // Nothing live -> no leak finding for this site.
        assert!(!f.iter().any(|f| f.detector == "monotonic-growth" && f.site == 2));
    }

    #[test]
    fn realloc_thrash_needs_many_grows() {
        let mut events = Vec::new();
        // 40 grow-copies of a growing buffer at site 3.
        for i in 0..40u64 {
            events.push(ev(EventKind::ReallocGrow, 0x100 + i, 64 * (i + 1), i, 3));
        }
        let rec = rec_with(&[(3, "Vec<u8>")], events);
        let f = analyze(&rec);
        assert!(f.iter().any(|f| f.detector == "realloc-thrash" && f.site == 3));
    }

    fn site_with_frames(id: u32, label: &str, frames: Vec<FrameMeta>) -> (u32, SiteInfo) {
        (id, SiteInfo { label: label.to_string(), frames })
    }

    fn frame(func: &str, file: &str, line: u32) -> FrameMeta {
        FrameMeta { func: func.to_string(), file: file.to_string(), line }
    }

    #[test]
    fn unrolled_sites_sharing_a_source_line_merge() {
        // Two distinct site ids whose boundary frame is the SAME source line
        // (the loop-unrolling case) must collapse into one finding.
        let mut sites = HashMap::new();
        let boundary = frame("app::work", "app.rs", 12);
        let [(i1, s1), (i2, s2)] = [
            site_with_frames(1, "Vec<u8>", vec![boundary.clone()]),
            site_with_frames(2, "Vec<u8>", vec![boundary.clone()]),
        ];
        sites.insert(i1, s1);
        sites.insert(i2, s2);
        let mut events = Vec::new();
        for i in 0..5_000u64 {
            events.push(ev(EventKind::Alloc, 0x10 + i, 1024, i * 2, 1));
            events.push(ev(EventKind::Dealloc, 0x10 + i, 1024, i * 2 + 1, 0));
            events.push(ev(EventKind::Alloc, 0x90000 + i, 1024, i * 2, 2));
            events.push(ev(EventKind::Dealloc, 0x90000 + i, 1024, i * 2 + 1, 0));
        }
        let rec = Recording { sites, meta: HashMap::new(), marks: HashMap::new(), events };
        let f = analyze(&rec);
        let churn: Vec<_> = f.iter().filter(|f| f.detector == "churn-storm").collect();
        assert_eq!(churn.len(), 1, "two unrolled sites should merge into one finding");
        assert_eq!(churn[0].location, "app::work (app.rs:12)");
    }

    #[test]
    fn profiler_origin_allocations_are_dropped() {
        // An allocation whose innermost non-std frame is memscope's own code is
        // measurement overhead — it must not appear as a finding.
        let mut sites = HashMap::new();
        let (id, info) = site_with_frames(
            5,
            "Vec<u8>",
            vec![frame("memscope_symbols::TypeOracle::build", "symbols.rs", 9)],
        );
        sites.insert(id, info);
        let mut events = Vec::new();
        for i in 0..5_000u64 {
            events.push(ev(EventKind::Alloc, 0x10 + i, 1024, i, 5));
        }
        let rec = Recording { sites, meta: HashMap::new(), marks: HashMap::new(), events };
        assert!(analyze(&rec).is_empty(), "profiler-origin site must be filtered out");
    }

    #[test]
    fn normal_vec_doubling_is_not_thrash() {
        // ~13 reallocations for 5000 pushes (doubling) — below the threshold.
        let mut events = Vec::new();
        for i in 0..13u64 {
            events.push(ev(EventKind::ReallocGrow, 0x200 + i, 64 * (i + 1), i, 4));
        }
        let rec = rec_with(&[(4, "Vec<u8>")], events);
        let f = analyze(&rec);
        assert!(!f.iter().any(|f| f.detector == "realloc-thrash"));
    }
}
