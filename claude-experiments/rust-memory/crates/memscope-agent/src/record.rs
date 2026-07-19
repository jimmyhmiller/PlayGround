//! Streams the full allocation event stream to a file — compact binary
//! (`.mscope`, default) or newline-JSON (`.json`/`.jsonl`, human-readable).
//!
//! Either way it's an [`EventSink`](memscope_core::EventSink) on the recorder's
//! pump (off the hot path). Sites are written **raw** — the interned id plus the
//! captured return addresses (`TAG_RSITE`) — the first time each is seen; the
//! header carries the load slide. **No DWARF runs in the traced process**:
//! symbolication (frames + recovered type + shape) is deferred to the reader
//! (`memscope replay`/`analyze`), which resolves the addresses against the
//! binary's dSYM once. This keeps a profiled program lean and fast even under a
//! heavy, multi-threaded allocation load, where building and querying the DWARF
//! index in-process previously dominated both memory and CPU.
//!
//! Per-event cost is the point: the binary writer encodes a fixed 34-byte record
//! into a reused buffer and flushes a whole batch in one write — ~10× smaller and
//! much faster than JSON, suitable for allocation-heavy programs.

use std::collections::HashSet;
use std::io::{BufWriter, Write};

use memscope_core::{self as mem, EventKind, EventSink, RawEvent};
use memscope_proto::{recfmt, SiteId};

enum Format {
    Binary,
    Json,
}

/// An [`EventSink`] that appends the event stream to a file.
///
/// Sites are written **raw** — just the interned id plus the captured return
/// addresses — so the hot consumer path never touches DWARF. Symbolication (type
/// recovery + stacks) happens in the reader (`memscope replay`/`analyze`), which
/// resolves the addresses against the binary's dSYM once, off the traced
/// process. The header carries the load slide so the reader can map runtime
/// addresses back to static ones.
pub struct FileRecorder {
    w: BufWriter<std::fs::File>,
    seen_sites: HashSet<u32>,
    seen_meta: HashSet<u32>,
    seen_keys: HashSet<u32>,
    seen_marks: HashSet<u32>,
    format: Format,
    scratch: Vec<u8>,
    written: u64,
}

impl FileRecorder {
    /// Create a recorder for `path`. Binary unless the path ends in `.json` or
    /// `.jsonl`. Builds the DWARF oracle once so sites resolve to types + stacks.
    pub fn create(path: &str) -> std::io::Result<Self> {
        let format = if path.ends_with(".json") || path.ends_with(".jsonl") {
            Format::Json
        } else {
            Format::Binary
        };
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);
        // The image memscope lives in — the main executable, OR a dylib (e.g. a
        // Node native addon) when injected. The frames + slide are relative to
        // *this* image, so the reader must symbolicate against it, not the host
        // executable (which would be `node` for an addon).
        let exe = memscope_symbols::current_image_path()
            .or_else(|| std::env::current_exe().ok())
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        // The load slide lets the reader map recorded runtime addresses back to
        // static (link-time) addresses for symbolication.
        let slide = memscope_symbols::current_image_slide();

        match format {
            Format::Binary => {
                let mut hdr = Vec::new();
                recfmt::encode_header(&mut hdr, std::process::id(), &exe, slide);
                w.write_all(&hdr)?;
            }
            Format::Json => {
                writeln!(
                    w,
                    "{{\"v\":2,\"pid\":{},\"exe\":{},\"slide\":{}}}",
                    std::process::id(),
                    json_str(&exe),
                    slide
                )?;
            }
        }

        Ok(FileRecorder {
            w,
            seen_sites: HashSet::new(),
            seen_meta: HashSet::new(),
            seen_keys: HashSet::new(),
            seen_marks: HashSet::new(),
            format,
            scratch: Vec::with_capacity(64 * 1024),
            written: 0,
        })
    }

    pub fn events_written(&self) -> u64 {
        self.written
    }

    /// Write a raw site definition (interned id + captured return addresses) the
    /// first time the site is seen. No DWARF here — the reader symbolicates.
    fn write_rsite(&mut self, site: u32) -> std::io::Result<()> {
        if !self.seen_sites.insert(site) {
            return Ok(());
        }
        let ips = mem::site_frames(SiteId(site)).unwrap_or_default();
        self.scratch.clear();
        recfmt::encode_rsite(&mut self.scratch, site, &ips);
        self.w.write_all(&self.scratch)
    }

    /// Write the `TAG_KEY` / `TAG_META` tables for a metadata context the first
    /// time it's referenced, so the file is self-contained.
    fn write_meta_def(&mut self, meta_id: u32) -> std::io::Result<()> {
        if !self.seen_meta.insert(meta_id) {
            return Ok(());
        }
        let kvs = match mem::meta_context(meta_id) {
            Some(kvs) => kvs,
            None => return Ok(()),
        };
        for (kid, _) in &kvs {
            if self.seen_keys.insert(*kid) {
                let name = mem::key_name(*kid).unwrap_or_default();
                self.scratch.clear();
                recfmt::encode_key(&mut self.scratch, *kid, &name);
                self.w.write_all(&self.scratch)?;
            }
        }
        self.scratch.clear();
        recfmt::encode_meta(&mut self.scratch, meta_id, &kvs);
        self.w.write_all(&self.scratch)?;
        Ok(())
    }

    /// Write the `TAG_MARK` label definition the first time a checkpoint is seen.
    fn write_mark_def(&mut self, label_id: u32) -> std::io::Result<()> {
        if !self.seen_marks.insert(label_id) {
            return Ok(());
        }
        let label = mem::mark_label(label_id).unwrap_or_default();
        self.scratch.clear();
        recfmt::encode_mark(&mut self.scratch, label_id, &label);
        self.w.write_all(&self.scratch)
    }

    fn consume_binary(&mut self, events: &[RawEvent]) -> std::io::Result<()> {
        // Emit any new site / metadata definitions first (so events can
        // reference them). For alloc events `site` is a stack site; for
        // MetaEnter it's a metadata context id.
        for e in events {
            match e.kind {
                EventKind::MetaEnter => {
                    self.write_meta_def(e.site.0)?;
                }
                EventKind::Mark => {
                    self.write_mark_def(e.site.0)?;
                }
                EventKind::Dealloc | EventKind::MetaExit => {}
                EventKind::Alloc | EventKind::ReallocGrow => {
                    if e.site.is_some() {
                        self.write_rsite(e.site.0)?;
                    }
                }
            }
        }
        // One event batch: header + fixed records, a single write.
        self.scratch.clear();
        recfmt::encode_events_header(&mut self.scratch, events.len() as u32);
        for e in events {
            recfmt::encode_event(&mut self.scratch, e);
        }
        self.w.write_all(&self.scratch)?;
        Ok(())
    }

    fn consume_json(&mut self, events: &[RawEvent]) -> std::io::Result<()> {
        for e in events {
            match e.kind {
                EventKind::Alloc | EventKind::ReallocGrow => {
                    if e.site.is_some() && self.seen_sites.insert(e.site.0) {
                        // Raw site: just the captured return addresses. The reader
                        // symbolicates against the binary's dSYM.
                        let ips = mem::site_frames(SiteId(e.site.0)).unwrap_or_default();
                        write!(self.w, "{{\"rsite\":{},\"ips\":[", e.site.0)?;
                        for (i, ip) in ips.iter().enumerate() {
                            if i > 0 {
                                write!(self.w, ",")?;
                            }
                            write!(self.w, "{ip}")?;
                        }
                        writeln!(self.w, "]}}")?;
                    }
                    let k = if e.kind == EventKind::Alloc { 'A' } else { 'R' };
                    writeln!(
                        self.w,
                        "{{\"k\":\"{k}\",\"ts\":{},\"a\":{},\"sz\":{},\"al\":{},\"s\":{},\"t\":{}}}",
                        e.ts_nanos, e.addr, e.size, e.align, e.site.0, e.thread
                    )?;
                }
                EventKind::Dealloc => {
                    writeln!(
                        self.w,
                        "{{\"k\":\"D\",\"ts\":{},\"a\":{},\"sz\":{},\"al\":{},\"t\":{}}}",
                        e.ts_nanos, e.addr, e.size, e.align, e.thread
                    )?;
                }
                EventKind::MetaEnter => {
                    if self.seen_meta.insert(e.site.0) {
                        if let Some(kvs) = mem::meta_context(e.site.0) {
                            write!(self.w, "{{\"meta\":{},\"kv\":{{", e.site.0)?;
                            for (i, (kid, val)) in kvs.iter().enumerate() {
                                let name = mem::key_name(*kid).unwrap_or_default();
                                if i > 0 {
                                    write!(self.w, ",")?;
                                }
                                write!(self.w, "{}:{}", json_str(&name), json_str(&val.to_display()))?;
                            }
                            writeln!(self.w, "}}}}")?;
                        }
                    }
                    writeln!(
                        self.w,
                        "{{\"k\":\"M\",\"ts\":{},\"s\":{},\"t\":{}}}",
                        e.ts_nanos, e.site.0, e.thread
                    )?;
                }
                EventKind::MetaExit => {
                    writeln!(
                        self.w,
                        "{{\"k\":\"m\",\"ts\":{},\"s\":{},\"t\":{}}}",
                        e.ts_nanos, e.site.0, e.thread
                    )?;
                }
                EventKind::Mark => {
                    if self.seen_marks.insert(e.site.0) {
                        let label = mem::mark_label(e.site.0).unwrap_or_default();
                        writeln!(
                            self.w,
                            "{{\"mark_def\":{},\"label\":{}}}",
                            e.site.0,
                            json_str(&label)
                        )?;
                    }
                    writeln!(
                        self.w,
                        "{{\"k\":\"MK\",\"ts\":{},\"s\":{},\"t\":{}}}",
                        e.ts_nanos, e.site.0, e.thread
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl EventSink for FileRecorder {
    fn consume(&mut self, events: &[RawEvent]) {
        let r = match self.format {
            Format::Binary => self.consume_binary(events),
            Format::Json => self.consume_json(events),
        };
        if r.is_err() {
            return;
        }
        self.written += events.len() as u64;
        let _ = self.w.flush();
    }

    fn flush(&mut self) {
        let _ = self.w.flush();
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Start recording the allocation stream to `path` (compact binary `.mscope` by
/// default; `.json`/`.jsonl` for human-readable). Switches the ring to Reliable
/// so nothing is dropped. Read it back with `memscope replay <file>`.
pub fn record_to_file(path: &str) -> std::io::Result<()> {
    let rec = FileRecorder::create(path)?;
    mem::set_ring_mode(mem::RingMode::Reliable);
    mem::spawn_consumer(Box::new(rec), std::time::Duration::from_millis(1));
    eprintln!("[memscope] recording allocations to {path}");
    Ok(())
}
