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

use std::collections::HashMap;

use memscope_proto::{recfmt, AllocShape, EventKind};

mod analysis;
mod frames;
pub use analysis::{analyze, site_stats, Finding, SiteStats};
pub use frames::{
    boundary_frame, clean_frame, frame_location, is_profiler_frame, is_profiler_origin,
    is_std_frame,
};

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

/// A parsed recording: per-site info, per-context metadata, mark labels, and the
/// ordered event stream.
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
    pub events: Vec<RecEvent>,
    /// Unresolved sites: interned id -> raw return addresses. Populated by
    /// [`read_recording_raw`]; empty once resolved into `sites`.
    pub raw_sites: HashMap<u32, Vec<u64>>,
    /// Binary path + load slide for symbolicating `raw_sites` (read-time).
    pub exe: String,
    pub slide: u64,
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
            // Keep only the boundary frame (first non-runtime) — all `analyze`
            // and `diff` need beyond the type label.
            let boundary = boundary_frame(&frames).cloned();
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

/// Read a recording (binary `.mscope` or JSON) into sites, metadata, marks, and
/// the event stream. Sites are fully symbolicated (may be expensive for large
/// recordings — see [`read_recording_raw`] for the constant-memory path).
pub fn read_recording(file: &str) -> Result<Recording, String> {
    let mut rec = read_recording_raw(file)?;
    let raw = std::mem::take(&mut rec.raw_sites);
    let all: Vec<u32> = raw.keys().copied().collect();
    rec.raw_sites = raw;
    rec.resolve_sites(&all);
    Ok(rec)
}

/// Read a recording **without symbolicating sites** — `raw_sites`/`exe`/`slide`
/// are populated and `sites` is left empty. Callers resolve only what they need
/// via [`Recording::resolve_sites`]. This keeps memory bounded for huge
/// recordings (e.g. ranking allocation sites needs no symbols).
pub fn read_recording_raw(file: &str) -> Result<Recording, String> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    {
        let mut f = std::fs::File::open(file).map_err(|e| e.to_string())?;
        let _ = f.read(&mut magic);
    }
    if recfmt::is_binary(&magic) {
        read_recording_binary(file)
    } else {
        read_recording_json(file)
    }
}

fn read_recording_binary(file: &str) -> Result<Recording, String> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(file).map_err(|e| e.to_string())?);
    let mut b1 = [0u8; 1];
    let mut b2 = [0u8; 2];
    let mut b4 = [0u8; 4];
    let rd_str = |f: &mut std::io::BufReader<std::fs::File>| -> Option<String> {
        let mut l = [0u8; 2];
        f.read_exact(&mut l).ok()?;
        let n = u16::from_le_bytes(l) as usize;
        let mut s = vec![0u8; n];
        f.read_exact(&mut s).ok()?;
        Some(String::from_utf8_lossy(&s).into_owned())
    };
    if f.read_exact(&mut b4).is_err() || b4 != recfmt::MAGIC {
        return Err("not a memscope binary recording".into());
    }
    let _ = f.read_exact(&mut b2); // version
    let _ = f.read_exact(&mut b2); // flags
    let _ = f.read_exact(&mut b4); // pid
    let exe = rd_str(&mut f).unwrap_or_default();
    // v2+ carries the load slide (for read-time symbolication of raw sites).
    let mut b8 = [0u8; 8];
    let slide = if f.read_exact(&mut b8).is_ok() {
        u64::from_le_bytes(b8)
    } else {
        0
    };

    let mut labels: HashMap<u32, SiteInfo> = HashMap::new();
    // Raw sites (TAG_RSITE): interned id -> captured return addresses, resolved
    // against the binary's dSYM after the stream is read (off the hot path).
    let mut raw_sites: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut events: Vec<RecEvent> = Vec::new();
    let mut key_names: HashMap<u32, String> = HashMap::new();
    let mut meta: HashMap<u32, Vec<(String, String)>> = HashMap::new();
    let mut marks: HashMap<u32, String> = HashMap::new();
    while f.read_exact(&mut b1).is_ok() {
        match b1[0] {
            recfmt::TAG_KEY => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let id = u32::from_le_bytes(b4);
                let name = rd_str(&mut f).unwrap_or_default();
                key_names.insert(id, name);
            }
            recfmt::TAG_META => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let id = u32::from_le_bytes(b4);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let mut kvs = Vec::new();
                for _ in 0..u16::from_le_bytes(b2) {
                    f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let kid = u32::from_le_bytes(b4);
                    let val = read_meta_value(&mut f).unwrap_or_default();
                    let key = key_names.get(&kid).cloned().unwrap_or_else(|| kid.to_string());
                    kvs.push((key, val));
                }
                meta.insert(id, kvs);
            }
            recfmt::TAG_MARK => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let id = u32::from_le_bytes(b4);
                let label = rd_str(&mut f).unwrap_or_default();
                marks.insert(id, label);
            }
            recfmt::TAG_SITE => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let site = u32::from_le_bytes(b4);
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let ty = if b1[0] == 1 { rd_str(&mut f) } else { None };
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let shape = recfmt::shape_from_code(b1[0]);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let mut frames = Vec::new();
                for _ in 0..u16::from_le_bytes(b2) {
                    let func = rd_str(&mut f).unwrap_or_default();
                    let file = rd_str(&mut f).unwrap_or_default();
                    f.read_exact(&mut b4).ok();
                    let line = u32::from_le_bytes(b4);
                    let _ = f.read_exact(&mut b1); // inlined
                    frames.push(FrameMeta { func, file, line });
                }
                labels.insert(
                    site,
                    SiteInfo {
                        label: label_for(shape, ty),
                        frames,
                    },
                );
            }
            recfmt::TAG_RSITE => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let site = u32::from_le_bytes(b4);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let n = u16::from_le_bytes(b2) as usize;
                let mut ips = Vec::with_capacity(n);
                let mut b8 = [0u8; 8];
                for _ in 0..n {
                    f.read_exact(&mut b8).map_err(|e| e.to_string())?;
                    ips.push(u64::from_le_bytes(b8));
                }
                raw_sites.insert(site, ips);
            }
            recfmt::TAG_EVENTS => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let count = u32::from_le_bytes(b4);
                let mut rec = vec![0u8; recfmt::EVENT_BYTES * count as usize];
                f.read_exact(&mut rec).map_err(|e| e.to_string())?;
                let mut r = recfmt::Reader::new(&rec);
                for _ in 0..count {
                    let Some(e) = r.decode_event() else { break };
                    events.push(RecEvent {
                        kind: e.kind,
                        addr: e.addr,
                        size: e.size,
                        ts_nanos: e.ts_nanos,
                        site: e.site.0,
                        thread: e.thread,
                    });
                }
            }
            other => return Err(format!("corrupt recording: unknown tag {other}")),
        }
    }

    // Carry sites unresolved; the caller symbolicates the subset it needs (see
    // `Recording::resolve_sites`). Legacy `TAG_SITE` recordings already have
    // resolved entries in `labels`.
    Ok(Recording {
        sites: labels,
        meta,
        marks,
        events,
        raw_sites,
        exe,
        slide,
    })
}

fn read_recording_json(file: &str) -> Result<Recording, String> {
    use std::io::BufRead;
    let rdr = std::io::BufReader::new(std::fs::File::open(file).map_err(|e| e.to_string())?);
    let mut labels: HashMap<u32, SiteInfo> = HashMap::new();
    let mut raw_sites: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut events: Vec<RecEvent> = Vec::new();
    let mut meta: HashMap<u32, Vec<(String, String)>> = HashMap::new();
    let mut marks: HashMap<u32, String> = HashMap::new();
    let mut exe = String::new();
    let mut slide = 0u64;
    for line in rdr.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("v").is_some() {
            // Header: {"v":2,"pid":…,"exe":…,"slide":…}
            exe = v.get("exe").and_then(|x| x.as_str()).unwrap_or("").to_string();
            slide = v.get("slide").and_then(|x| x.as_u64()).unwrap_or(0);
            continue;
        }
        // Raw site: {"rsite":id,"ips":[…]} — resolved after the stream is read.
        if let Some(id) = v.get("rsite").and_then(|x| x.as_u64()) {
            let ips = v
                .get("ips")
                .and_then(|x| x.as_array())
                .map(|arr| arr.iter().filter_map(|n| n.as_u64()).collect())
                .unwrap_or_default();
            raw_sites.insert(id as u32, ips);
            continue;
        }
        // Mark label definition: {"mark_def":id,"label":"…"}
        if let Some(id) = v.get("mark_def").and_then(|x| x.as_u64()) {
            let label = v.get("label").and_then(|x| x.as_str()).unwrap_or("").to_string();
            marks.insert(id as u32, label);
            continue;
        }
        // Metadata context definition: {"meta":id,"kv":{k:v,...}}
        if let Some(id) = v.get("meta").and_then(|x| x.as_u64()) {
            if let Some(obj) = v.get("kv").and_then(|x| x.as_object()) {
                let kvs = obj
                    .iter()
                    .map(|(k, val)| {
                        let vs = match val {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k.clone(), vs)
                    })
                    .collect();
                meta.insert(id as u32, kvs);
            }
            continue;
        }
        if let Some(site) = v.get("site").and_then(|x| x.as_u64()) {
            let ty = v.get("ty").and_then(|x| x.as_str()).map(String::from);
            let shape = v.get("shape").and_then(|x| x.as_str());
            let label = match (shape, &ty) {
                (Some(sh), Some(t)) => format!("{sh}<{t}>"),
                (Some(sh), None) => format!("{sh}<?>"),
                (None, Some(t)) => t.clone(),
                (None, None) => "<no type>".into(),
            };
            let frames = v
                .get("frames")
                .and_then(|f| f.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|fr| {
                            let a = fr.as_array()?;
                            Some(FrameMeta {
                                func: a.first()?.as_str()?.to_string(),
                                file: a.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                                line: a.get(2).and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            labels.insert(site as u32, SiteInfo { label, frames });
            continue;
        }
        let kind = match v.get("k").and_then(|x| x.as_str()) {
            Some("A") => EventKind::Alloc,
            Some("R") => EventKind::ReallocGrow,
            Some("D") => EventKind::Dealloc,
            Some("M") => EventKind::MetaEnter,
            Some("m") => EventKind::MetaExit,
            Some("MK") => EventKind::Mark,
            _ => continue,
        };
        events.push(RecEvent {
            kind,
            addr: v.get("a").and_then(|x| x.as_u64()).unwrap_or(0),
            size: v.get("sz").and_then(|x| x.as_u64()).unwrap_or(0),
            ts_nanos: v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0),
            site: v.get("s").and_then(|x| x.as_u64()).unwrap_or(u32::MAX as u64) as u32,
            thread: v.get("t").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        });
    }
    Ok(Recording {
        sites: labels,
        meta,
        marks,
        events,
        raw_sites,
        exe,
        slide,
    })
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
    /// Index into `Recording::events` where this mark sits.
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

/// The reconstructed live set at a point in the stream.
#[derive(Clone, Debug)]
pub struct LiveState {
    /// The checkpoint this state was taken at, or `None` for end-of-stream.
    pub at: Option<MarkPoint>,
    /// Exclusive end index in `Recording::events` that was replayed to here.
    pub upto: usize,
    pub live: Vec<LiveAlloc>,
    pub total_live_bytes: u64,
}

/// A view over a [`Recording`] that locates marks and reconstructs the live set
/// at any of them (or at end-of-stream).
pub struct Timeline<'a> {
    rec: &'a Recording,
    marks: Vec<MarkPoint>,
}

impl<'a> Timeline<'a> {
    /// Scan the event stream for mark checkpoints.
    pub fn new(rec: &'a Recording) -> Self {
        let mut marks = Vec::new();
        for (i, e) in rec.events.iter().enumerate() {
            if e.kind == EventKind::Mark {
                let label = rec.marks.get(&e.site).cloned().unwrap_or_else(|| e.site.to_string());
                marks.push(MarkPoint {
                    label,
                    label_id: e.site,
                    ts_nanos: e.ts_nanos,
                    index: i,
                });
            }
        }
        Timeline { rec, marks }
    }

    /// All checkpoints, in stream order.
    pub fn marks(&self) -> &[MarkPoint] {
        &self.marks
    }

    /// The first checkpoint with this label, if any. (Labels are usually unique;
    /// callers that re-`mark` the same label get the earliest occurrence.)
    pub fn find(&self, label: &str) -> Option<&MarkPoint> {
        self.marks.iter().find(|m| m.label == label)
    }

    /// Reconstruct the live set by replaying events `[0, upto)`.
    pub fn state_at_index(&self, upto: usize, at: Option<MarkPoint>) -> LiveState {
        let upto = upto.min(self.rec.events.len());
        let mut live: HashMap<u64, LiveAlloc> = HashMap::new();
        for e in &self.rec.events[..upto] {
            match e.kind {
                EventKind::Alloc | EventKind::ReallocGrow => {
                    live.insert(
                        e.addr,
                        LiveAlloc {
                            addr: e.addr,
                            size: e.size,
                            site: e.site,
                            ts_nanos: e.ts_nanos,
                            thread: e.thread,
                        },
                    );
                }
                EventKind::Dealloc => {
                    live.remove(&e.addr);
                }
                // Markers carry no allocation.
                EventKind::MetaEnter | EventKind::MetaExit | EventKind::Mark => {}
            }
        }
        let total_live_bytes = live.values().map(|a| a.size).sum();
        let mut live: Vec<LiveAlloc> = live.into_values().collect();
        live.sort_by_key(|a| a.addr);
        LiveState { at, upto, live, total_live_bytes }
    }

    /// The live set at a named checkpoint (inclusive of the mark's position),
    /// or `None` if no such mark exists.
    pub fn state_at(&self, label: &str) -> Option<LiveState> {
        let m = self.find(label)?.clone();
        // The mark itself is a no-op; replaying through its index is exact.
        Some(self.state_at_index(m.index + 1, Some(m)))
    }

    /// The live set at the end of the recording.
    pub fn state_at_end(&self) -> LiveState {
        self.state_at_index(self.rec.events.len(), None)
    }

    /// Per-site `(born, freed)` counts for allocations within the window
    /// `[from.upto, to_upto)`.
    ///
    /// Dealloc events carry no site id, so frees are attributed by seeding
    /// `addr -> site` from the window's start state (`from`) and tracking each
    /// allocation made through the window. A site with `born > 0 && freed == 0`
    /// is the canonical "born and never died" leak fingerprint.
    pub fn window_born_freed(&self, from: &LiveState, to_upto: usize) -> HashMap<u32, (u64, u64)> {
        let to_upto = to_upto.min(self.rec.events.len());
        let mut addr_site: HashMap<u64, u32> =
            from.live.iter().map(|a| (a.addr, a.site)).collect();
        // (born, freed) per site.
        let mut counts: HashMap<u32, (u64, u64)> = HashMap::new();
        let start = from.upto.min(to_upto);
        for e in &self.rec.events[start..to_upto] {
            match e.kind {
                EventKind::Alloc | EventKind::ReallocGrow => {
                    addr_site.insert(e.addr, e.site);
                    counts.entry(e.site).or_default().0 += 1;
                }
                EventKind::Dealloc => {
                    if let Some(s) = addr_site.remove(&e.addr) {
                        counts.entry(s).or_default().1 += 1;
                    }
                }
                EventKind::MetaEnter | EventKind::MetaExit | EventKind::Mark => {}
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(kind: EventKind, addr: u64, size: u64, ts: u64, site: u32) -> RecEvent {
        RecEvent { kind, addr, size, ts_nanos: ts, site, thread: 1 }
    }

    fn recording_with_marks() -> Recording {
        let mut sites = HashMap::new();
        sites.insert(7, SiteInfo { label: "Boxed<Particle>".into(), frames: vec![] });
        let mut marks = HashMap::new();
        marks.insert(0, "warmup".to_string());
        marks.insert(1, "end".to_string());
        let events = vec![
            ev(EventKind::Alloc, 0x10, 64, 1, 7),
            ev(EventKind::Alloc, 0x20, 64, 2, 7),
            ev(EventKind::Mark, 0, 0, 3, 0), // "warmup": 2 live (0x10,0x20)
            ev(EventKind::Dealloc, 0x10, 64, 4, 7),
            ev(EventKind::Alloc, 0x30, 128, 5, 7),
            ev(EventKind::Mark, 1, 0, 6, 1), // "end": 0x20,0x30 live = 192 bytes
        ];
        Recording { sites, meta: HashMap::new(), marks, events, ..Default::default() }
    }

    #[test]
    fn timeline_finds_marks_in_order() {
        let rec = recording_with_marks();
        let tl = Timeline::new(&rec);
        let labels: Vec<&str> = tl.marks().iter().map(|m| m.label.as_str()).collect();
        assert_eq!(labels, ["warmup", "end"]);
    }

    #[test]
    fn state_at_warmup_has_two_live() {
        let rec = recording_with_marks();
        let tl = Timeline::new(&rec);
        let s = tl.state_at("warmup").unwrap();
        assert_eq!(s.live.len(), 2);
        assert_eq!(s.total_live_bytes, 128);
    }

    #[test]
    fn state_at_end_reflects_free_and_new_alloc() {
        let rec = recording_with_marks();
        let tl = Timeline::new(&rec);
        let s = tl.state_at("end").unwrap();
        // 0x10 freed, 0x20 + 0x30 live.
        let addrs: Vec<u64> = s.live.iter().map(|a| a.addr).collect();
        assert_eq!(addrs, [0x20, 0x30]);
        assert_eq!(s.total_live_bytes, 192);
    }

    #[test]
    fn missing_mark_is_none() {
        let rec = recording_with_marks();
        let tl = Timeline::new(&rec);
        assert!(tl.state_at("nope").is_none());
    }

    #[test]
    fn window_born_freed_attributes_frees_by_seeded_site() {
        let rec = recording_with_marks();
        let tl = Timeline::new(&rec);
        let warmup = tl.state_at("warmup").unwrap();
        let end = tl.state_at("end").unwrap();
        // Window warmup->end: free 0x10 (site 7, seeded from warmup) + alloc 0x30 (site 7).
        let bf = tl.window_born_freed(&warmup, end.upto);
        assert_eq!(bf.get(&7).copied(), Some((1, 1)));
    }

    #[test]
    fn window_born_freed_flags_pure_leak() {
        // Two allocs after the start mark, never freed -> born=2, freed=0.
        let mut sites = HashMap::new();
        sites.insert(9, SiteInfo { label: "Leak".into(), frames: vec![] });
        let events = vec![
            ev(EventKind::Mark, 0, 0, 1, 0),
            ev(EventKind::Alloc, 0x10, 32, 2, 9),
            ev(EventKind::Alloc, 0x20, 32, 3, 9),
        ];
        let mut marks = HashMap::new();
        marks.insert(0, "start".to_string());
        let rec = Recording { sites, meta: HashMap::new(), marks, events, ..Default::default() };
        let tl = Timeline::new(&rec);
        let start = tl.state_at("start").unwrap();
        let bf = tl.window_born_freed(&start, rec.events.len());
        assert_eq!(bf.get(&9).copied(), Some((2, 0)));
    }
}
