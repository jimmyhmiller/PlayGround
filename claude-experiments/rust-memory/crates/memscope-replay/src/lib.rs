//! Reusable reader for memscope recordings.
//!
//! A recording (`.mscope` binary or `.json`/`.jsonl`) is **self-contained**:
//! resolved site types + stacks, metadata, and mark labels are embedded, so a
//! reader needs no binary or debug info. This crate parses one into a
//! [`Recording`] (sites, metadata, marks, and the ordered event stream) and
//! reconstructs the **live set at any checkpoint** via [`Timeline`] /
//! [`LiveState`].
//!
//! It's the shared substrate for posthoc tooling — `memscope diff`, `marks`, and
//! (later) `analyze` all build on the same reconstruction rather than each
//! re-implementing event replay.
//!
//! # Memory contract
//!
//! Everything here is **constant in the event count**. A [`Recording`] holds only
//! the recording's *definitions* (sites, metadata, mark labels) — its events are
//! never a field, only an [`EventStream`] you fold over once. Peak memory for any
//! analysis is `O(sites + live set)`, both of which are properties of the program
//! being measured, not of how long it ran.
//!
//! This matters because full-mode recordings of real programs are dominated by
//! short-lived allocation *churn*: a `next build` emits ~50M events, of which a
//! few hundred thousand are live at any instant. Materializing that stream cost
//! ~3× the recording size and OOM-killed the analyzers.

use std::collections::HashMap;

use memscope_proto::{AllocShape, EventKind};

mod analysis;
mod frames;
mod stream;
pub use analysis::{analyze, site_stats, Finding, SiteStats};
pub use frames::{
    boundary_frame, clean_frame, frame_location, is_profiler_frame, is_profiler_origin,
    is_std_frame,
};
pub use stream::{stream_events, EventStream, Record, RecordReader};

// --- parsed recording --------------------------------------------------------

/// A decoded event from a recording (either format). Stream order is causal
/// (applied) order — replaying front-to-back reconstructs the exact live set.
#[derive(Clone, Debug)]
pub struct RecEvent {
    pub kind: EventKind,
    pub addr: u64,
    pub size: u64,
    pub ts_nanos: u64,
    /// For Alloc/Realloc: the allocation site id. For Mark: the label id. For
    /// MetaEnter/MetaExit: the metadata context id.
    pub site: u32,
    pub thread: u32,
}

/// One resolved stack frame: function name + source location.
#[derive(Default, Clone, Debug)]
pub struct FrameMeta {
    pub func: String,
    pub file: String,
    pub line: u32,
}

/// Resolved info for one allocation site: its type label and its call stack
/// (innermost first — as recorded).
#[derive(Default, Clone, Debug)]
pub struct SiteInfo {
    pub label: String,
    pub frames: Vec<FrameMeta>,
}

/// A parsed recording's **definitions**: per-site info, per-context metadata, and
/// mark labels.
///
/// Deliberately *not* the events. Events are orders of magnitude more numerous
/// than definitions and are only ever consumed as a fold, so they're reached via
/// [`stream_events`] (or [`Timeline`], which layers live-set reconstruction on
/// top). A `Recording` is small enough to keep around; the stream is not.
///
/// Sites can be carried **unresolved** (`raw_sites` populated, `sites` empty)
/// when read via [`read_recording_raw`], so a caller can symbolicate only the
/// subset it needs — symbolicating *every* site of a large recording costs many
/// GB, but the numeric per-site stats (bytes/counts, which drive ranking) need
/// no symbols at all. See [`Recording::resolve_sites`].
#[derive(Default, Clone, Debug)]
pub struct Recording {
    pub sites: HashMap<u32, SiteInfo>,
    /// meta context id -> resolved [(key, value)] pairs.
    pub meta: HashMap<u32, Vec<(String, String)>>,
    /// mark label id -> human label (from `memscope::mark`).
    pub marks: HashMap<u32, String>,
    /// Unresolved sites: interned id -> raw return addresses. Populated by
    /// [`read_recording_raw`]; empty once resolved into `sites`.
    pub raw_sites: HashMap<u32, Vec<u64>>,
    /// Binary path + load slide for symbolicating `raw_sites` (read-time).
    pub exe: String,
    pub slide: u64,
    /// Pid of the recorded process.
    pub pid: u32,
}

impl Recording {
    /// The type label for a site id (e.g. `Boxed<Particle>`), or a placeholder.
    pub fn site_label(&self, site: u32) -> &str {
        self.sites.get(&site).map(|s| s.label.as_str()).unwrap_or("<unknown site>")
    }

    /// Symbolicate just the given site ids (from `raw_sites`) into `sites`. This
    /// is the constant-memory path: cost scales with `wanted.len()`, not with the
    /// total number of sites in the recording. No-op for ids already resolved or
    /// absent from `raw_sites`.
    pub fn resolve_sites(&mut self, wanted: &[u32]) {
        let pending: Vec<(u32, Vec<u64>)> = wanted
            .iter()
            .filter(|id| !self.sites.contains_key(id))
            .filter_map(|id| self.raw_sites.get(id).map(|ips| (*id, ips.clone())))
            .collect();
        if pending.is_empty() {
            return;
        }
        if let Ok(resolved) =
            memscope_symbols::resolve_raw_sites(std::path::Path::new(&self.exe), self.slide, &pending)
        {
            for (id, r) in resolved {
                let frames = r
                    .frames
                    .into_iter()
                    .map(|fr| FrameMeta {
                        func: fr.function.unwrap_or_default(),
                        file: fr.file.unwrap_or_default(),
                        line: fr.line.unwrap_or(0),
                    })
                    .collect();
                self.sites.insert(
                    id,
                    SiteInfo {
                        label: label_for(r.shape, r.element_type),
                        frames,
                    },
                );
            }
        }
    }
}

impl Recording {
    /// Symbolicate sites with **bounded memory**, keeping only a compact result
    /// per site (its type label + boundary frame) rather than every site's full
    /// stack. Resolves each *unique* IP once via [`memscope_symbols::SiteResolver`],
    /// then assembles each site transiently and discards its frames.
    ///
    /// This is what makes `analyze` work on recordings with millions of deep
    /// sites built from a few thousand IPs (full-frame resolution would need tens
    /// of GB). `analyze`/`diff` only need the label + boundary location, which is
    /// exactly what's retained.
    pub fn resolve_sites_compact(&mut self) {
        if self.raw_sites.is_empty() {
            return;
        }
        let unique: Vec<u64> = {
            let mut set: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for ips in self.raw_sites.values() {
                set.extend(ips.iter().copied());
            }
            set.into_iter().collect()
        };
        let resolver = match memscope_symbols::SiteResolver::build(
            std::path::Path::new(&self.exe),
            self.slide,
            &unique,
        ) {
            Ok(r) => r,
            Err(_) => return,
        };
        for (id, ips) in &self.raw_sites {
            let r = resolver.resolve_site(ips);
            let frames: Vec<FrameMeta> = r
                .frames
                .into_iter()
                .map(|fr| FrameMeta {
                    func: fr.function.unwrap_or_default(),
                    file: fr.file.unwrap_or_default(),
                    line: fr.line.unwrap_or(0),
                })
                .collect();
            // Keep only one frame: the boundary (first non-runtime), or the
            // innermost frame when the whole stack is runtime — which is the
            // fallback `site_loc` reports, so a pure-runtime site still gets a
            // location instead of an empty string. Callers that ask for the
            // boundary still get `None` for those, since the retained frame is
            // itself a runtime frame.
            let boundary = boundary_frame(&frames).or_else(|| frames.first()).cloned();
            self.sites.insert(
                *id,
                SiteInfo {
                    label: label_for(r.shape, r.element_type),
                    frames: boundary.into_iter().collect(),
                },
            );
        }
    }
}

/// Read a recording's definitions (binary `.mscope` or JSON) with sites fully
/// symbolicated. Events are **not** read — fold over [`stream_events`] for those.
///
/// Symbolicating every site can be expensive on a recording with many distinct
/// sites; see [`read_recording_raw`] + [`Recording::resolve_sites_compact`] for
/// the bounded path.
pub fn read_recording(file: &str) -> Result<Recording, String> {
    let mut rec = read_recording_raw(file)?;
    let all: Vec<u32> = rec.raw_sites.keys().copied().collect();
    rec.resolve_sites(&all);
    Ok(rec)
}

/// Read a recording's definitions **without symbolicating sites** —
/// `raw_sites`/`exe`/`slide` are populated and `sites` is left empty. Callers
/// resolve only what they need via [`Recording::resolve_sites`].
///
/// Event payloads are seeked past, not decoded, so this costs one scan of the
/// definition records regardless of how many events the recording holds.
pub fn read_recording_raw(file: &str) -> Result<Recording, String> {
    let mut r = RecordReader::open(file)?.skipping_events();
    let mut rec = Recording::default();
    while let Some(record) = r.next_record()? {
        match record {
            Record::Site(id, info) => {
                rec.sites.insert(id, info);
            }
            Record::RawSite(id, ips) => {
                rec.raw_sites.insert(id, ips);
            }
            Record::Meta(id, kvs) => {
                rec.meta.insert(id, kvs);
            }
            Record::MarkLabel(id, label) => {
                rec.marks.insert(id, label);
            }
            // `skipping_events` means these never arrive.
            Record::Event(_) => {}
        }
    }
    rec.exe = r.exe().to_string();
    rec.slide = r.slide();
    rec.pid = r.pid();
    Ok(rec)
}

/// Read one encoded `MetaValue` from a stream, returning its display string
/// (also used to skip the bytes when the value isn't needed).
pub fn read_meta_value(f: &mut impl std::io::Read) -> Option<String> {
    let mut t = [0u8; 1];
    f.read_exact(&mut t).ok()?;
    let mut b8 = [0u8; 8];
    match t[0] {
        0 => {
            let mut l = [0u8; 2];
            f.read_exact(&mut l).ok()?;
            let n = u16::from_le_bytes(l) as usize;
            let mut s = vec![0u8; n];
            f.read_exact(&mut s).ok()?;
            Some(String::from_utf8_lossy(&s).into_owned())
        }
        1 => {
            f.read_exact(&mut b8).ok()?;
            Some(i64::from_le_bytes(b8).to_string())
        }
        2 => {
            f.read_exact(&mut b8).ok()?;
            Some(u64::from_le_bytes(b8).to_string())
        }
        3 => {
            f.read_exact(&mut b8).ok()?;
            Some(f64::from_bits(u64::from_le_bytes(b8)).to_string())
        }
        4 => {
            let mut b1 = [0u8; 1];
            f.read_exact(&mut b1).ok()?;
            Some((b1[0] != 0).to_string())
        }
        _ => None,
    }
}

/// Build a site's type label from its recovered shape + element type.
pub fn label_for(shape: Option<AllocShape>, ty: Option<String>) -> String {
    match (shape, ty) {
        (Some(sh), Some(t)) => format!("{sh:?}<{t}>"),
        (Some(sh), None) => format!("{sh:?}<?>"),
        (None, Some(t)) => t,
        (None, None) => "<no type>".into(),
    }
}

// --- timeline & live-set reconstruction --------------------------------------

/// A named checkpoint located in the event stream (`memscope::mark`).
#[derive(Clone, Debug)]
pub struct MarkPoint {
    pub label: String,
    pub label_id: u32,
    pub ts_nanos: u64,
    /// Position of this mark in the event stream.
    pub index: usize,
}

/// One live allocation in a reconstructed [`LiveState`].
#[derive(Clone, Copy, Debug)]
pub struct LiveAlloc {
    pub addr: u64,
    pub size: u64,
    pub site: u32,
    pub ts_nanos: u64,
    pub thread: u32,
}

/// The set of allocations live at a point in the stream, keyed by address.
///
/// This is the one structure a replay is allowed to grow: it holds the *live
/// set*, which is what you're trying to inspect, and drops freed allocations as
/// it goes. Feed it events in causal order with [`LiveSet::apply`].
#[derive(Default, Clone, Debug)]
pub struct LiveSet {
    map: HashMap<u64, LiveAlloc>,
    bytes: u64,
}

impl LiveSet {
    pub fn new() -> LiveSet {
        LiveSet::default()
    }

    /// Apply one event, inserting or removing a live allocation.
    ///
    /// Returns the allocation that **left** the live set, if any — the one a
    /// `Dealloc` freed, or the one an address-reusing allocation displaced.
    /// Frees carry no site id of their own, so this is how a caller attributes
    /// one back to the site that allocated it without keeping a second
    /// `addr -> site` map alongside.
    pub fn apply(&mut self, e: &RecEvent) -> Option<LiveAlloc> {
        match e.kind {
            EventKind::Alloc | EventKind::ReallocGrow => {
                let prev = self.map.insert(
                    e.addr,
                    LiveAlloc {
                        addr: e.addr,
                        size: e.size,
                        site: e.site,
                        ts_nanos: e.ts_nanos,
                        thread: e.thread,
                    },
                );
                if let Some(p) = &prev {
                    self.bytes = self.bytes.saturating_sub(p.size);
                }
                self.bytes += e.size;
                prev
            }
            EventKind::Dealloc => {
                let prev = self.map.remove(&e.addr);
                if let Some(p) = &prev {
                    self.bytes = self.bytes.saturating_sub(p.size);
                }
                prev
            }
            // Markers carry no allocation.
            EventKind::MetaEnter | EventKind::MetaExit | EventKind::Mark => None,
        }
    }

    /// Total bytes currently live.
    pub fn total_bytes(&self) -> u64 {
        self.bytes
    }

    /// Number of live allocations.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The live allocations, in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &LiveAlloc> {
        self.map.values()
    }

    /// The site of the allocation live at `addr`, if any.
    pub fn site_of(&self, addr: u64) -> Option<u32> {
        self.map.get(&addr).map(|a| a.site)
    }

    /// Aggregate by site id -> `(count, bytes)`.
    pub fn agg_by_site(&self) -> HashMap<u32, (u64, u64)> {
        let mut m: HashMap<u32, (u64, u64)> = HashMap::new();
        for a in self.map.values() {
            let e = m.entry(a.site).or_default();
            e.0 += 1;
            e.1 += a.size;
        }
        m
    }

    /// Snapshot as a sorted `Vec` (address order).
    pub fn to_sorted_vec(&self) -> Vec<LiveAlloc> {
        let mut v: Vec<LiveAlloc> = self.map.values().copied().collect();
        v.sort_by_key(|a| a.addr);
        v
    }
}

/// The reconstructed live set at a point in the stream.
#[derive(Clone, Debug)]
pub struct LiveState {
    /// The checkpoint this state was taken at, or `None` for end-of-stream.
    pub at: Option<MarkPoint>,
    /// Exclusive end index in the event stream that was replayed to here.
    pub upto: usize,
    pub live: Vec<LiveAlloc>,
    pub total_live_bytes: u64,
}

/// What one scan of a recording's event stream learns about its shape: where the
/// checkpoints are, how many events there are, and when the last one happened.
///
/// All of it is `O(marks)` — a recording with 50M events yields a `StreamIndex`
/// of a few hundred bytes.
#[derive(Default, Clone, Debug)]
pub struct StreamIndex {
    pub marks: Vec<MarkPoint>,
    pub event_count: usize,
    pub last_ts: u64,
}

/// A view over a recording that locates marks and reconstructs the live set at
/// any of them (or at end-of-stream).
///
/// Each reconstruction re-streams the file rather than indexing into a retained
/// event list — trading a re-read (sequential, page-cache friendly) for memory
/// that doesn't scale with the recording. Prefer the methods that answer your
/// whole question in one pass ([`Timeline::for_each_mark`], [`Timeline::window`])
/// over calling [`Timeline::state_at`] in a loop.
pub struct Timeline<'a> {
    rec: &'a Recording,
    file: String,
    index: StreamIndex,
}

impl<'a> Timeline<'a> {
    /// Scan the event stream once for checkpoints, event count, and end time.
    pub fn open(file: &str, rec: &'a Recording) -> Result<Timeline<'a>, String> {
        let mut index = StreamIndex::default();
        let mut stream = stream_events(file)?;
        for (i, e) in stream.by_ref().enumerate() {
            index.event_count = i + 1;
            index.last_ts = index.last_ts.max(e.ts_nanos);
            if e.kind == EventKind::Mark {
                let label =
                    rec.marks.get(&e.site).cloned().unwrap_or_else(|| e.site.to_string());
                index.marks.push(MarkPoint {
                    label,
                    label_id: e.site,
                    ts_nanos: e.ts_nanos,
                    index: i,
                });
            }
        }
        if let Some(err) = stream.error() {
            return Err(err.to_string());
        }
        Ok(Timeline { rec, file: file.to_string(), index })
    }

    /// All checkpoints, in stream order.
    pub fn marks(&self) -> &[MarkPoint] {
        &self.index.marks
    }

    /// Total events in the recording.
    pub fn event_count(&self) -> usize {
        self.index.event_count
    }

    /// Timestamp of the last event.
    pub fn last_ts(&self) -> u64 {
        self.index.last_ts
    }

    /// The recording whose definitions back this timeline.
    pub fn recording(&self) -> &'a Recording {
        self.rec
    }

    /// The first checkpoint with this label, if any. (Labels are usually unique;
    /// callers that re-`mark` the same label get the earliest occurrence.)
    pub fn find(&self, label: &str) -> Option<&MarkPoint> {
        self.index.marks.iter().find(|m| m.label == label)
    }

    /// Stream the events, calling `f` after each one is applied, and return the
    /// live set at the point the scan stopped.
    ///
    /// The building block the other methods share: `f` sees the live set as of
    /// event `i`, plus whatever allocation that event removed (see
    /// [`LiveSet::apply`]), and can accumulate whatever aggregate it needs.
    /// Returning `false` stops the scan early.
    pub fn replay<F>(&self, mut f: F) -> Result<LiveSet, String>
    where
        F: FnMut(usize, &RecEvent, Option<&LiveAlloc>, &LiveSet) -> bool,
    {
        let mut live = LiveSet::new();
        let mut stream = stream_events(&self.file)?;
        for (i, e) in stream.by_ref().enumerate() {
            let removed = live.apply(&e);
            if !f(i, &e, removed.as_ref(), &live) {
                return Ok(live);
            }
        }
        match stream.error() {
            Some(err) => Err(err.to_string()),
            None => Ok(live),
        }
    }

    /// Reconstruct the live set by replaying events `[0, upto)`. An `upto` past
    /// the end of the stream yields the final state.
    pub fn state_at_index(&self, upto: usize, at: Option<MarkPoint>) -> Result<LiveState, String> {
        if upto == 0 {
            return Ok(LiveState { at, upto, live: Vec::new(), total_live_bytes: 0 });
        }
        let mut snapshot: Option<LiveSet> = None;
        let final_live = self.replay(|i, _e, _removed, live| {
            if i + 1 == upto {
                snapshot = Some(live.clone());
                return false;
            }
            true
        })?;
        let live = snapshot.unwrap_or(final_live);
        let total_live_bytes = live.total_bytes();
        Ok(LiveState { at, upto, live: live.to_sorted_vec(), total_live_bytes })
    }

    /// The live set at a named checkpoint (inclusive of the mark's position), or
    /// `None` if no such mark exists.
    pub fn state_at(&self, label: &str) -> Result<Option<LiveState>, String> {
        let Some(m) = self.find(label).cloned() else { return Ok(None) };
        // The mark itself is a no-op; replaying through its index is exact.
        self.state_at_index(m.index + 1, Some(m)).map(Some)
    }

    /// The live set at the end of the recording.
    pub fn state_at_end(&self) -> Result<LiveState, String> {
        self.state_at_index(self.index.event_count, None)
    }

    /// Stream once, invoking `f` at every checkpoint with the live set as of that
    /// mark.
    ///
    /// This is how to summarize *all* the marks: calling [`Timeline::state_at`]
    /// per mark would re-stream the file once per mark.
    pub fn for_each_mark<F>(&self, mut f: F) -> Result<(), String>
    where
        F: FnMut(&MarkPoint, &LiveSet),
    {
        let mut next = 0usize;
        self.replay(|i, e, _removed, live| {
            // Marks were indexed in stream order, so this walks them in step.
            if e.kind == EventKind::Mark {
                if let Some(m) = self.index.marks.get(next).filter(|m| m.index == i) {
                    f(m, live);
                    next += 1;
                }
            }
            true
        })?;
        Ok(())
    }

    /// Everything a two-checkpoint comparison needs, from **one** pass over the
    /// stream: the live set aggregated by site at each endpoint, and per-site
    /// born/freed counts within the window `[a_upto, b_upto)`.
    ///
    /// Frees carry no site id, so they're attributed via the allocation the live
    /// set drops (see [`LiveSet::apply`]) — exact for anything the replay saw
    /// allocated, including allocations born before `a_upto`.
    pub fn window(&self, a_upto: usize, b_upto: usize) -> Result<Window, String> {
        let a_upto = a_upto.min(b_upto);
        let mut w = Window::default();
        let (mut got_a, mut got_b) = (a_upto == 0, false);
        let final_live = self.replay(|i, e, removed, live| {
            let n = i + 1; // events applied so far
            if n > a_upto && n <= b_upto {
                match e.kind {
                    EventKind::Alloc | EventKind::ReallocGrow => {
                        w.born_freed.entry(e.site).or_default().0 += 1;
                    }
                    EventKind::Dealloc => {
                        if let Some(freed) = removed {
                            w.born_freed.entry(freed.site).or_default().1 += 1;
                        }
                    }
                    _ => {}
                }
            }
            if n == a_upto {
                w.a_by_site = live.agg_by_site();
                w.a_bytes = live.total_bytes();
                got_a = true;
            }
            if n == b_upto {
                w.b_by_site = live.agg_by_site();
                w.b_bytes = live.total_bytes();
                got_b = true;
                return false;
            }
            true
        })?;
        // An endpoint at or past end-of-stream never hit its `n ==` check; the
        // stream ran out first, so the final state is what it meant.
        if !got_b {
            w.b_by_site = final_live.agg_by_site();
            w.b_bytes = final_live.total_bytes();
        }
        if !got_a {
            w.a_by_site = w.b_by_site.clone();
            w.a_bytes = w.b_bytes;
        }
        Ok(w)
    }
}

/// The result of comparing two points in a stream — see [`Timeline::window`].
#[derive(Default, Clone, Debug)]
pub struct Window {
    /// Live set at endpoint A, aggregated site -> `(count, bytes)`.
    pub a_by_site: HashMap<u32, (u64, u64)>,
    pub a_bytes: u64,
    /// Live set at endpoint B, aggregated site -> `(count, bytes)`.
    pub b_by_site: HashMap<u32, (u64, u64)>,
    pub b_bytes: u64,
    /// Per-site `(born, freed)` within the window. A site with `born > 0 &&
    /// freed == 0` is the canonical "born and never died" leak fingerprint.
    pub born_freed: HashMap<u32, (u64, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use memscope_proto::{recfmt, RawEvent, SiteId};

    fn ev(kind: EventKind, addr: u64, size: u64, ts: u64, site: u32) -> RecEvent {
        RecEvent { kind, addr, size, ts_nanos: ts, site, thread: 1 }
    }

    /// Write a real `.mscope` file so the tests exercise the streaming reader
    /// rather than a hand-built in-memory struct.
    struct TempRec {
        path: std::path::PathBuf,
    }

    impl TempRec {
        fn write(name: &str, marks: &[(u32, &str)], events: &[RecEvent]) -> TempRec {
            let mut b = Vec::new();
            recfmt::encode_header(&mut b, 1, "/tmp/test-exe", 0);
            for (id, label) in marks {
                recfmt::encode_mark(&mut b, *id, label);
            }
            recfmt::encode_events_header(&mut b, events.len() as u32);
            for e in events {
                recfmt::encode_event(
                    &mut b,
                    &RawEvent {
                        kind: e.kind,
                        addr: e.addr,
                        size: e.size,
                        ts_nanos: e.ts_nanos,
                        site: SiteId(e.site),
                        thread: e.thread,
                        align: 8,
                        seq: 0,
                    },
                );
            }
            let path = std::env::temp_dir().join(format!("memscope-replay-test-{name}.mscope"));
            std::fs::write(&path, &b).expect("write test recording");
            TempRec { path }
        }

        fn file(&self) -> &str {
            self.path.to_str().unwrap()
        }
    }

    impl Drop for TempRec {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn events_with_marks() -> Vec<RecEvent> {
        vec![
            ev(EventKind::Alloc, 0x10, 64, 1, 7),
            ev(EventKind::Alloc, 0x20, 64, 2, 7),
            ev(EventKind::Mark, 0, 0, 3, 0), // "warmup": 2 live (0x10,0x20)
            ev(EventKind::Dealloc, 0x10, 64, 4, 7),
            ev(EventKind::Alloc, 0x30, 128, 5, 7),
            ev(EventKind::Mark, 1, 0, 6, 1), // "end": 0x20,0x30 live = 192 bytes
        ]
    }

    fn recording_with_marks(name: &str) -> (TempRec, Recording) {
        let tmp = TempRec::write(name, &[(0, "warmup"), (1, "end")], &events_with_marks());
        let mut rec = read_recording_raw(tmp.file()).expect("read recording");
        rec.sites.insert(7, SiteInfo { label: "Boxed<Particle>".into(), frames: vec![] });
        (tmp, rec)
    }

    #[test]
    fn reader_round_trips_marks_and_events() {
        let (tmp, rec) = recording_with_marks("roundtrip");
        assert_eq!(rec.marks.get(&0).map(String::as_str), Some("warmup"));
        assert_eq!(rec.exe, "/tmp/test-exe");
        let streamed: Vec<RecEvent> = stream_events(tmp.file()).unwrap().collect();
        assert_eq!(streamed.len(), 6);
        assert_eq!(streamed[0].addr, 0x10);
        assert_eq!(streamed[4].size, 128);
    }

    #[test]
    fn timeline_finds_marks_in_order() {
        let (tmp, rec) = recording_with_marks("marks-order");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let labels: Vec<&str> = tl.marks().iter().map(|m| m.label.as_str()).collect();
        assert_eq!(labels, ["warmup", "end"]);
        assert_eq!(tl.event_count(), 6);
        assert_eq!(tl.last_ts(), 6);
    }

    #[test]
    fn state_at_warmup_has_two_live() {
        let (tmp, rec) = recording_with_marks("warmup");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let s = tl.state_at("warmup").unwrap().unwrap();
        assert_eq!(s.live.len(), 2);
        assert_eq!(s.total_live_bytes, 128);
    }

    #[test]
    fn state_at_end_reflects_free_and_new_alloc() {
        let (tmp, rec) = recording_with_marks("end");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let s = tl.state_at("end").unwrap().unwrap();
        // 0x10 freed, 0x20 + 0x30 live.
        let addrs: Vec<u64> = s.live.iter().map(|a| a.addr).collect();
        assert_eq!(addrs, [0x20, 0x30]);
        assert_eq!(s.total_live_bytes, 192);
        // Replaying past the end of the stream means "the final state".
        let past = tl.state_at_index(usize::MAX, None).unwrap();
        assert_eq!(past.total_live_bytes, 192);
    }

    #[test]
    fn missing_mark_is_none() {
        let (tmp, rec) = recording_with_marks("missing");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        assert!(tl.state_at("nope").unwrap().is_none());
    }

    #[test]
    fn for_each_mark_visits_every_checkpoint_in_one_pass() {
        let (tmp, rec) = recording_with_marks("each-mark");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let mut seen: Vec<(String, u64)> = Vec::new();
        tl.for_each_mark(|m, live| seen.push((m.label.clone(), live.total_bytes()))).unwrap();
        assert_eq!(
            seen,
            vec![("warmup".to_string(), 128u64), ("end".to_string(), 192u64)]
        );
    }

    #[test]
    fn window_attributes_frees_by_the_allocation_that_left() {
        let (tmp, rec) = recording_with_marks("window");
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let a = tl.find("warmup").unwrap().index + 1;
        let b = tl.find("end").unwrap().index + 1;
        // Window warmup->end: free 0x10 (site 7, allocated before the window) +
        // alloc 0x30 (site 7).
        let w = tl.window(a, b).unwrap();
        assert_eq!(w.born_freed.get(&7).copied(), Some((1, 1)));
        assert_eq!(w.a_bytes, 128);
        assert_eq!(w.b_bytes, 192);
        assert_eq!(w.a_by_site.get(&7).copied(), Some((2, 128)));
        assert_eq!(w.b_by_site.get(&7).copied(), Some((2, 192)));
    }

    #[test]
    fn window_flags_pure_leak() {
        // Two allocs after the start mark, never freed -> born=2, freed=0.
        let events = vec![
            ev(EventKind::Mark, 0, 0, 1, 0),
            ev(EventKind::Alloc, 0x10, 32, 2, 9),
            ev(EventKind::Alloc, 0x20, 32, 3, 9),
        ];
        let tmp = TempRec::write("leak", &[(0, "start")], &events);
        let rec = read_recording_raw(tmp.file()).unwrap();
        let tl = Timeline::open(tmp.file(), &rec).unwrap();
        let start = tl.find("start").unwrap().index + 1;
        let w = tl.window(start, tl.event_count()).unwrap();
        assert_eq!(w.born_freed.get(&9).copied(), Some((2, 0)));
        assert_eq!(w.b_bytes, 64);
    }

    #[test]
    fn live_set_charges_address_reuse_exactly() {
        // An address reallocated without an intervening free replaces the entry;
        // total bytes must reflect the new size, not the sum.
        let mut live = LiveSet::new();
        live.apply(&ev(EventKind::Alloc, 0x10, 64, 1, 1));
        let displaced = live.apply(&ev(EventKind::ReallocGrow, 0x10, 256, 2, 1));
        assert_eq!(displaced.map(|a| a.size), Some(64));
        assert_eq!(live.total_bytes(), 256);
        assert_eq!(live.len(), 1);
        let freed = live.apply(&ev(EventKind::Dealloc, 0x10, 256, 3, 0));
        assert_eq!(freed.map(|a| a.site), Some(1));
        assert_eq!(live.total_bytes(), 0);
    }
}
