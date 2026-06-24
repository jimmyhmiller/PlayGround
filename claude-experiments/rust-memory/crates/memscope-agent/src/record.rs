//! Streams the full allocation event stream to a self-contained file.
//!
//! This is an [`EventSink`](memscope_core::EventSink) that runs on the recorder's
//! pump (off the hot path). It writes newline-delimited JSON so the result is
//! trivially parseable in any language — open it, read it line by line, build
//! whatever viewer you like.
//!
//! The file is **self-contained**: the first time an allocation site is seen it
//! resolves the site (stack frames + recovered Rust type + container shape) via
//! the binary's DWARF and writes a one-off `site` record. Every allocation /
//! free is then a compact `event` record referencing that site id. A reader
//! never needs the original binary or its debug info.
//!
//! Format (one JSON object per line):
//! ```text
//! {"v":1,"pid":1234,"exe":"/path/to/bin"}                         // header (first line)
//! {"site":37,"ty":"Particle","shape":"Boxed","frames":[["serve::main","serve.rs",53,false], ...]}
//! {"k":"A","ts":2434000000,"a":41923141376,"sz":64,"al":8,"s":37,"t":2}   // alloc
//! {"k":"D","ts":2440000000,"a":41923141376,"sz":64,"al":8,"t":2}          // free
//! ```
//! `k` = A(lloc) / D(ealloc) / R(ealloc-grow); `a` addr, `sz` size, `al` align,
//! `s` site id, `t` thread, `ts` ns since start.

use std::collections::HashSet;
use std::io::{BufWriter, Write};

use memscope_core::{self as mem, EventKind, EventSink, RawEvent};
use memscope_proto::SiteId;
use memscope_symbols::TypeOracle;

/// An [`EventSink`] that appends the resolved event stream to a file.
pub struct FileRecorder {
    w: BufWriter<std::fs::File>,
    oracle: Option<TypeOracle>,
    seen_sites: HashSet<u32>,
    /// JSON-escaped scratch reused across writes.
    written: u64,
}

impl FileRecorder {
    /// Create a recorder writing to `path`. Builds the DWARF oracle once so the
    /// file can carry resolved types + stacks.
    pub fn create(path: &str) -> std::io::Result<Self> {
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);
        let exe = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        writeln!(
            w,
            "{{\"v\":1,\"pid\":{},\"exe\":{}}}",
            std::process::id(),
            json_str(&exe)
        )?;
        let oracle = TypeOracle::for_current_process().ok();
        Ok(FileRecorder {
            w,
            oracle,
            seen_sites: HashSet::new(),
            written: 0,
        })
    }

    pub fn events_written(&self) -> u64 {
        self.written
    }

    fn write_site(&mut self, site: u32) -> std::io::Result<()> {
        if !self.seen_sites.insert(site) {
            return Ok(());
        }
        let ips = mem::site_frames(SiteId(site)).unwrap_or_default();
        let (frames, ty, shape) = match &self.oracle {
            Some(o) => o.resolve_site_ips(&ips),
            None => (Vec::new(), None, None),
        };
        write!(self.w, "{{\"site\":{site}")?;
        if let Some(t) = &ty {
            write!(self.w, ",\"ty\":{}", json_str(t))?;
        }
        if let Some(s) = shape {
            write!(self.w, ",\"shape\":\"{s:?}\"")?;
        }
        write!(self.w, ",\"frames\":[")?;
        for (i, fr) in frames.iter().enumerate() {
            if i > 0 {
                write!(self.w, ",")?;
            }
            write!(
                self.w,
                "[{},{},{},{}]",
                json_str(fr.function.as_deref().unwrap_or("")),
                json_str(fr.file.as_deref().unwrap_or("")),
                fr.line.unwrap_or(0),
                fr.inlined
            )?;
        }
        writeln!(self.w, "]}}")?;
        Ok(())
    }

    fn write_event(&mut self, e: &RawEvent) -> std::io::Result<()> {
        let k = match e.kind {
            EventKind::Alloc => 'A',
            EventKind::Dealloc => 'D',
            EventKind::ReallocGrow => 'R',
        };
        if e.kind == EventKind::Dealloc {
            writeln!(
                self.w,
                "{{\"k\":\"{k}\",\"ts\":{},\"a\":{},\"sz\":{},\"al\":{},\"t\":{}}}",
                e.ts_nanos, e.addr, e.size, e.align, e.thread
            )
        } else {
            if e.site.is_some() {
                self.write_site(e.site.0)?;
            }
            writeln!(
                self.w,
                "{{\"k\":\"{k}\",\"ts\":{},\"a\":{},\"sz\":{},\"al\":{},\"s\":{},\"t\":{}}}",
                e.ts_nanos, e.addr, e.size, e.align, e.site.0, e.thread
            )
        }
    }
}

impl EventSink for FileRecorder {
    fn consume(&mut self, events: &[RawEvent]) {
        for e in events {
            // Best-effort: a write error (disk full, closed) just stops output.
            if self.write_event(e).is_err() {
                return;
            }
            self.written += 1;
        }
        let _ = self.w.flush();
    }

    fn flush(&mut self) {
        let _ = self.w.flush();
    }
}

/// Minimal JSON string escaping (paths/symbols rarely need more than this).
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

/// Start recording the allocation stream to `path`. Switches the ring to
/// Reliable mode (so nothing is dropped) and installs the file recorder on the
/// pump. Returns the number of recorded events is tracked on the sink; the
/// recording runs until the process exits.
pub fn record_to_file(path: &str) -> std::io::Result<()> {
    let rec = FileRecorder::create(path)?;
    // Don't lose events while recording a complete trace.
    mem::set_ring_mode(mem::RingMode::Reliable);
    mem::spawn_consumer(Box::new(rec), std::time::Duration::from_millis(1));
    eprintln!("[memscope] recording allocations to {path}");
    Ok(())
}
