//! Streaming record reader — the one place either recording format is parsed.
//!
//! A recording is a sequence of **records**: site definitions, metadata/mark
//! definitions, and allocation events. This module walks that sequence lazily so
//! no consumer ever has to materialize the event stream.
//!
//! That is the memory contract of the whole crate: a recording holds *tens of
//! millions* of events (a `next build` produces ~2 GB of them), but every
//! analysis over one is a fold — peak memory is `O(sites + live set)`, never
//! `O(events)`. Reading the events into a `Vec` was worth ~3× the recording size
//! and OOM-killed the analyzers, so the events simply are not addressable as a
//! collection: you get an [`Iterator`], once, in causal order.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use memscope_proto::{recfmt, EventKind};

use crate::{label_for, FrameMeta, RecEvent, SiteInfo};

/// 1 MiB of read-ahead: event blocks are dense, and the default 8 KiB buffer
/// costs a syscall every ~240 events.
const READ_BUF: usize = 1 << 20;

/// One record from a recording, in file order.
#[derive(Debug)]
pub enum Record {
    /// A pre-resolved site definition (legacy `TAG_SITE` recordings).
    Site(u32, SiteInfo),
    /// An unresolved site: interned id -> captured return addresses.
    RawSite(u32, Vec<u64>),
    /// A metadata context: id -> resolved key/value pairs.
    Meta(u32, Vec<(String, String)>),
    /// A mark label definition: id -> human label.
    MarkLabel(u32, String),
    /// One allocation-stream event.
    Event(RecEvent),
}

/// A lazy cursor over a recording's records, in causal order.
///
/// Construct with [`RecordReader::open`]. With [`RecordReader::skipping_events`]
/// the event payload is seeked past rather than decoded — that's the cheap pass
/// used to read a recording's header (sites/marks/metadata) without paying to
/// decode millions of events you're going to drop anyway.
pub struct RecordReader {
    inner: Inner,
    /// Binary path recorded in the header, for symbolication.
    exe: String,
    /// Main-image ASLR load slide at record time.
    slide: u64,
    /// Pid of the recorded process.
    pid: u32,
    skip_events: bool,
}

enum Inner {
    Bin(Bin),
    Json(Json),
}

impl RecordReader {
    /// Open a recording (binary `.mscope` or JSON `.jsonl`), detecting the format
    /// from its magic.
    pub fn open(file: &str) -> Result<RecordReader, String> {
        let mut magic = [0u8; 4];
        {
            let mut f = std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?;
            let _ = f.read(&mut magic);
        }
        let f = std::fs::File::open(file).map_err(|e| format!("{file}: {e}"))?;
        if recfmt::is_binary(&magic) {
            let (bin, exe, slide, pid) = Bin::open(f)?;
            Ok(RecordReader { inner: Inner::Bin(bin), exe, slide, pid, skip_events: false })
        } else {
            let json = Json::new(BufReader::with_capacity(READ_BUF, f));
            Ok(RecordReader {
                inner: Inner::Json(json),
                exe: String::new(),
                slide: 0,
                pid: 0,
                skip_events: false,
            })
        }
    }

    /// Seek past event payloads instead of decoding them. Only definition records
    /// are yielded.
    pub fn skipping_events(mut self) -> Self {
        self.skip_events = true;
        self
    }

    /// The recorded binary path (populated after the header is parsed; for JSON
    /// that is once the header line has been read).
    pub fn exe(&self) -> &str {
        &self.exe
    }

    /// The recorded ASLR load slide.
    pub fn slide(&self) -> u64 {
        self.slide
    }

    /// The pid of the recorded process.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The next record, or `None` at end of stream.
    pub fn next_record(&mut self) -> Result<Option<Record>, String> {
        match &mut self.inner {
            Inner::Bin(b) => b.next_record(self.skip_events),
            Inner::Json(j) => {
                let r = j.next_record(self.skip_events)?;
                if self.exe.is_empty() && !j.exe.is_empty() {
                    self.exe = j.exe.clone();
                    self.slide = j.slide;
                    self.pid = j.pid;
                }
                Ok(r)
            }
        }
    }
}

// --- binary format -----------------------------------------------------------

struct Bin {
    f: BufReader<std::fs::File>,
    /// Events remaining in the current `TAG_EVENTS` block.
    pending: u32,
    /// Interned metadata key names, needed to resolve `TAG_META` records.
    key_names: HashMap<u32, String>,
}

impl Bin {
    fn open(f: std::fs::File) -> Result<(Bin, String, u64, u32), String> {
        let mut f = BufReader::with_capacity(READ_BUF, f);
        let mut b2 = [0u8; 2];
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4).map_err(|e| e.to_string())?;
        if b4 != recfmt::MAGIC {
            return Err("not a memscope binary recording".into());
        }
        let _ = f.read_exact(&mut b2); // version
        let _ = f.read_exact(&mut b2); // flags
        let _ = f.read_exact(&mut b4);
        let pid = u32::from_le_bytes(b4);
        let exe = read_str(&mut f).unwrap_or_default();
        // v2+ carries the load slide (for read-time symbolication of raw sites).
        let mut b8 = [0u8; 8];
        let slide = if f.read_exact(&mut b8).is_ok() { u64::from_le_bytes(b8) } else { 0 };
        Ok((Bin { f, pending: 0, key_names: HashMap::new() }, exe, slide, pid))
    }

    fn next_record(&mut self, skip_events: bool) -> Result<Option<Record>, String> {
        loop {
            if self.pending > 0 {
                self.pending -= 1;
                let mut buf = [0u8; recfmt::EVENT_BYTES];
                self.f.read_exact(&mut buf).map_err(|e| e.to_string())?;
                let e = recfmt::Reader::new(&buf)
                    .decode_event()
                    .ok_or("corrupt recording: truncated event")?;
                return Ok(Some(Record::Event(RecEvent {
                    kind: e.kind,
                    addr: e.addr,
                    size: e.size,
                    ts_nanos: e.ts_nanos,
                    site: e.site.0,
                    thread: e.thread,
                })));
            }

            let mut b1 = [0u8; 1];
            if self.f.read_exact(&mut b1).is_err() {
                return Ok(None); // clean end of stream
            }
            let mut b2 = [0u8; 2];
            let mut b4 = [0u8; 4];
            match b1[0] {
                recfmt::TAG_KEY => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let id = u32::from_le_bytes(b4);
                    let name = read_str(&mut self.f).unwrap_or_default();
                    self.key_names.insert(id, name);
                }
                recfmt::TAG_META => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let id = u32::from_le_bytes(b4);
                    self.f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                    let mut kvs = Vec::new();
                    for _ in 0..u16::from_le_bytes(b2) {
                        self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                        let kid = u32::from_le_bytes(b4);
                        let val = crate::read_meta_value(&mut self.f).unwrap_or_default();
                        let key =
                            self.key_names.get(&kid).cloned().unwrap_or_else(|| kid.to_string());
                        kvs.push((key, val));
                    }
                    return Ok(Some(Record::Meta(id, kvs)));
                }
                recfmt::TAG_MARK => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let id = u32::from_le_bytes(b4);
                    let label = read_str(&mut self.f).unwrap_or_default();
                    return Ok(Some(Record::MarkLabel(id, label)));
                }
                recfmt::TAG_SITE => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let site = u32::from_le_bytes(b4);
                    self.f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                    let ty = if b1[0] == 1 { read_str(&mut self.f) } else { None };
                    self.f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                    let shape = recfmt::shape_from_code(b1[0]);
                    self.f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                    let mut frames = Vec::new();
                    for _ in 0..u16::from_le_bytes(b2) {
                        let func = read_str(&mut self.f).unwrap_or_default();
                        let file = read_str(&mut self.f).unwrap_or_default();
                        self.f.read_exact(&mut b4).ok();
                        let line = u32::from_le_bytes(b4);
                        let _ = self.f.read_exact(&mut b1); // inlined
                        frames.push(FrameMeta { func, file, line });
                    }
                    return Ok(Some(Record::Site(
                        site,
                        SiteInfo { label: label_for(shape, ty), frames },
                    )));
                }
                recfmt::TAG_RSITE => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let site = u32::from_le_bytes(b4);
                    self.f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                    let n = u16::from_le_bytes(b2) as usize;
                    let mut ips = Vec::with_capacity(n);
                    let mut b8 = [0u8; 8];
                    for _ in 0..n {
                        self.f.read_exact(&mut b8).map_err(|e| e.to_string())?;
                        ips.push(u64::from_le_bytes(b8));
                    }
                    return Ok(Some(Record::RawSite(site, ips)));
                }
                recfmt::TAG_EVENTS => {
                    self.f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let count = u32::from_le_bytes(b4);
                    if skip_events {
                        let bytes = recfmt::EVENT_BYTES as i64 * count as i64;
                        self.f.seek(SeekFrom::Current(bytes)).map_err(|e| e.to_string())?;
                    } else {
                        self.pending = count;
                    }
                }
                other => return Err(format!("corrupt recording: unknown tag {other}")),
            }
        }
    }
}

fn read_str(f: &mut BufReader<std::fs::File>) -> Option<String> {
    let mut l = [0u8; 2];
    f.read_exact(&mut l).ok()?;
    let n = u16::from_le_bytes(l) as usize;
    let mut s = vec![0u8; n];
    f.read_exact(&mut s).ok()?;
    Some(String::from_utf8_lossy(&s).into_owned())
}

// --- JSON format -------------------------------------------------------------

struct Json {
    f: BufReader<std::fs::File>,
    line: String,
    exe: String,
    slide: u64,
    pid: u32,
}

impl Json {
    fn new(f: BufReader<std::fs::File>) -> Json {
        Json { f, line: String::new(), exe: String::new(), slide: 0, pid: 0 }
    }

    fn next_record(&mut self, skip_events: bool) -> Result<Option<Record>, String> {
        loop {
            self.line.clear();
            let n = self.f.read_line(&mut self.line).map_err(|e| e.to_string())?;
            if n == 0 {
                return Ok(None);
            }
            let line = self.line.trim_end();
            // Event lines are by far the most common; when we're skipping them,
            // recognizing one by prefix avoids parsing the JSON at all.
            if skip_events && line.starts_with("{\"k\":") {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("v").is_some() {
                self.exe = v.get("exe").and_then(|x| x.as_str()).unwrap_or("").to_string();
                self.slide = v.get("slide").and_then(|x| x.as_u64()).unwrap_or(0);
                self.pid = v.get("pid").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                continue;
            }
            if let Some(id) = v.get("rsite").and_then(|x| x.as_u64()) {
                let ips = v
                    .get("ips")
                    .and_then(|x| x.as_array())
                    .map(|arr| arr.iter().filter_map(|n| n.as_u64()).collect())
                    .unwrap_or_default();
                return Ok(Some(Record::RawSite(id as u32, ips)));
            }
            if let Some(id) = v.get("mark_def").and_then(|x| x.as_u64()) {
                let label = v.get("label").and_then(|x| x.as_str()).unwrap_or("").to_string();
                return Ok(Some(Record::MarkLabel(id as u32, label)));
            }
            if let Some(id) = v.get("meta").and_then(|x| x.as_u64()) {
                let kvs = v
                    .get("kv")
                    .and_then(|x| x.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, val)| {
                                let vs = match val {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                (k.clone(), vs)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                return Ok(Some(Record::Meta(id as u32, kvs)));
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
                                    file: a
                                        .get(1)
                                        .and_then(|x| x.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    line: a.get(2).and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                return Ok(Some(Record::Site(site as u32, SiteInfo { label, frames })));
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
            return Ok(Some(Record::Event(RecEvent {
                kind,
                addr: v.get("a").and_then(|x| x.as_u64()).unwrap_or(0),
                size: v.get("sz").and_then(|x| x.as_u64()).unwrap_or(0),
                ts_nanos: v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0),
                site: v.get("s").and_then(|x| x.as_u64()).unwrap_or(u32::MAX as u64) as u32,
                thread: v.get("t").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            })));
        }
    }
}

// --- the event stream --------------------------------------------------------

/// A lazy iterator over a recording's events, in causal (applied) order.
///
/// This is the **only** way to see a recording's events, and it is single-pass by
/// construction: folding over it is `O(1)` in the event count. Definition records
/// encountered along the way are skipped — read those once up front with
/// [`crate::read_recording_raw`].
///
/// A malformed record ends the stream (the events already yielded are still
/// valid); [`EventStream::error`] reports it afterwards.
pub struct EventStream {
    reader: RecordReader,
    error: Option<String>,
    done: bool,
}

impl EventStream {
    /// The parse error that ended the stream early, if any.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

impl Iterator for EventStream {
    type Item = RecEvent;

    fn next(&mut self) -> Option<RecEvent> {
        if self.done {
            return None;
        }
        loop {
            match self.reader.next_record() {
                Ok(Some(Record::Event(e))) => return Some(e),
                Ok(Some(_)) => continue,
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.error = Some(e);
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

/// Stream a recording's events without materializing them.
///
/// ```no_run
/// // Total bytes allocated, in constant memory, over a 2 GB recording.
/// let total: u64 = memscope_replay::stream_events("big.mscope")?
///     .filter(|e| e.kind == memscope_proto::EventKind::Alloc)
///     .map(|e| e.size)
///     .sum();
/// # Ok::<(), String>(())
/// ```
pub fn stream_events(file: &str) -> Result<EventStream, String> {
    Ok(EventStream { reader: RecordReader::open(file)?, error: None, done: false })
}
