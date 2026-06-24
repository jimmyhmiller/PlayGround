//! Compact binary recording format (`.mscope`) — written in real time.
//!
//! The hot consumer path writes a fixed **34-byte** record per event (no JSON,
//! no per-event allocation; a whole batch is encoded into a reused buffer and
//! flushed in one `write_all`). Sites are resolved once each (interned — even a
//! billion-allocation run has only a handful of distinct sites) and written as a
//! one-off `site` record, so the file stays **self-contained**: a reader needs
//! no binary or debug info.
//!
//! Layout (little-endian):
//! ```text
//! header:  magic "MSCP" | u16 version | u16 flags | u32 pid | u16 exe_len | exe…
//! record:  u8 tag, then:
//!   'S' site:   u32 id | u8 has_ty (+ u16 len+utf8) | u8 shape_code | u16 nframes
//!               nframes × [u16 fn_len+utf8 | u16 file_len+utf8 | u32 line | u8 inlined]
//!   'E' events: u32 count | count × [u8 kind | u64 addr | u64 size | u64 ts
//!               | u32 site | u32 thread | u8 align_log2]   (34 bytes each)
//! ```

use crate::{AllocShape, EventKind, RawEvent, SiteId};

pub const MAGIC: [u8; 4] = *b"MSCP";
pub const VERSION: u16 = 1;
pub const TAG_SITE: u8 = b'S';
pub const TAG_EVENTS: u8 = b'E';
/// Size of one encoded event record (the per-event hot-path cost).
pub const EVENT_BYTES: usize = 1 + 8 + 8 + 8 + 4 + 4 + 1;

const SHAPE_NONE: u8 = 0;
pub fn shape_to_code(s: Option<AllocShape>) -> u8 {
    match s {
        None => SHAPE_NONE,
        Some(AllocShape::Boxed) => 1,
        Some(AllocShape::Vec) => 2,
        Some(AllocShape::Rc) => 3,
        Some(AllocShape::Arc) => 4,
        Some(AllocShape::HashTable) => 5,
        Some(AllocShape::StringBuf) => 6,
        Some(AllocShape::Other) => 7,
    }
}
pub fn shape_from_code(c: u8) -> Option<AllocShape> {
    match c {
        1 => Some(AllocShape::Boxed),
        2 => Some(AllocShape::Vec),
        3 => Some(AllocShape::Rc),
        4 => Some(AllocShape::Arc),
        5 => Some(AllocShape::HashTable),
        6 => Some(AllocShape::StringBuf),
        7 => Some(AllocShape::Other),
        _ => None,
    }
}

fn kind_code(k: EventKind) -> u8 {
    match k {
        EventKind::Alloc => 0,
        EventKind::Dealloc => 1,
        EventKind::ReallocGrow => 2,
    }
}
fn kind_from_code(c: u8) -> EventKind {
    match c {
        1 => EventKind::Dealloc,
        2 => EventKind::ReallocGrow,
        _ => EventKind::Alloc,
    }
}

// --- writer helpers (append to a reused buffer) ------------------------------

fn put_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    put_u16(b, bytes.len().min(u16::MAX as usize) as u16);
    b.extend_from_slice(&bytes[..bytes.len().min(u16::MAX as usize)]);
}

/// Encode the file header into `b`.
pub fn encode_header(b: &mut Vec<u8>, pid: u32, exe: &str) {
    b.extend_from_slice(&MAGIC);
    put_u16(b, VERSION);
    put_u16(b, 0); // flags
    put_u32(b, pid);
    put_str(b, exe);
}

/// A resolved frame as stored in a site record.
pub struct FrameRec<'a> {
    pub function: &'a str,
    pub file: &'a str,
    pub line: u32,
    pub inlined: bool,
}

/// Encode a one-off site definition (resolved frames + type + shape) into `b`.
pub fn encode_site(
    b: &mut Vec<u8>,
    site: u32,
    ty: Option<&str>,
    shape: Option<AllocShape>,
    frames: &[FrameRec],
) {
    b.push(TAG_SITE);
    put_u32(b, site);
    match ty {
        Some(t) => {
            b.push(1);
            put_str(b, t);
        }
        None => b.push(0),
    }
    b.push(shape_to_code(shape));
    put_u16(b, frames.len().min(u16::MAX as usize) as u16);
    for f in frames.iter().take(u16::MAX as usize) {
        put_str(b, f.function);
        put_str(b, f.file);
        put_u32(b, f.line);
        b.push(f.inlined as u8);
    }
}

/// Begin an event batch of `count` events; follow with `encode_event` × count.
pub fn encode_events_header(b: &mut Vec<u8>, count: u32) {
    b.push(TAG_EVENTS);
    put_u32(b, count);
}

/// Encode one event (34 bytes). The order key (`seq`) is dropped — the file is
/// already written in applied (causal) order.
#[inline]
pub fn encode_event(b: &mut Vec<u8>, e: &RawEvent) {
    b.push(kind_code(e.kind));
    put_u64(b, e.addr);
    put_u64(b, e.size);
    put_u64(b, e.ts_nanos);
    put_u32(b, e.site.0);
    put_u32(b, e.thread);
    b.push(align_log2(e.align));
}

#[inline]
fn align_log2(align: u32) -> u8 {
    align.max(1).trailing_zeros() as u8
}

// --- reader ------------------------------------------------------------------

/// A decoded event from a recording.
#[derive(Clone, Copy, Debug)]
pub struct DecodedEvent {
    pub kind: EventKind,
    pub addr: u64,
    pub size: u64,
    pub ts_nanos: u64,
    pub site: SiteId,
    pub thread: u32,
    pub align: u32,
}

/// A cursor over a byte slice for decoding.
pub struct Reader<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    pub fn u8(&mut self) -> Option<u8> {
        let v = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(v)
    }
    pub fn u16(&mut self) -> Option<u16> {
        let b = self.buf.get(self.pos..self.pos + 2)?;
        self.pos += 2;
        Some(u16::from_le_bytes(b.try_into().ok()?))
    }
    pub fn u32(&mut self) -> Option<u32> {
        let b = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    }
    pub fn u64(&mut self) -> Option<u64> {
        let b = self.buf.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        Some(u64::from_le_bytes(b.try_into().ok()?))
    }
    pub fn str(&mut self) -> Option<String> {
        let n = self.u16()? as usize;
        let b = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(String::from_utf8_lossy(b).into_owned())
    }

    pub fn decode_event(&mut self) -> Option<DecodedEvent> {
        let kind = kind_from_code(self.u8()?);
        let addr = self.u64()?;
        let size = self.u64()?;
        let ts_nanos = self.u64()?;
        let site = SiteId(self.u32()?);
        let thread = self.u32()?;
        let align = 1u32 << self.u8()?;
        Some(DecodedEvent {
            kind,
            addr,
            size,
            ts_nanos,
            site,
            thread,
            align,
        })
    }
}

/// Parsed header.
pub struct Header {
    pub version: u16,
    pub pid: u32,
    pub exe: String,
}

/// Validate magic + parse the header; returns the header and bytes consumed.
pub fn decode_header(buf: &[u8]) -> Option<(Header, usize)> {
    let mut r = Reader::new(buf);
    let magic = [r.u8()?, r.u8()?, r.u8()?, r.u8()?];
    if magic != MAGIC {
        return None;
    }
    let version = r.u16()?;
    let _flags = r.u16()?;
    let pid = r.u32()?;
    let exe = r.str()?;
    Some((Header { version, pid, exe }, r.pos))
}

/// Quick check: does this look like a binary `.mscope` recording?
pub fn is_binary(buf: &[u8]) -> bool {
    buf.len() >= 4 && buf[..4] == MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let mut b = Vec::new();
        encode_header(&mut b, 4242, "/path/to/bin");
        assert!(is_binary(&b));
        let (h, used) = decode_header(&b).unwrap();
        assert_eq!(h.version, VERSION);
        assert_eq!(h.pid, 4242);
        assert_eq!(h.exe, "/path/to/bin");
        assert_eq!(used, b.len());
    }

    #[test]
    fn event_roundtrip() {
        let mut b = Vec::new();
        let ev = RawEvent {
            kind: EventKind::Alloc,
            seq: 99,
            ts_nanos: 123_456_789,
            addr: 0xdead_beef,
            size: 4096,
            align: 16,
            site: SiteId(37),
            thread: 7,
        };
        encode_event(&mut b, &ev);
        assert_eq!(b.len(), EVENT_BYTES);
        let d = Reader::new(&b).decode_event().unwrap();
        assert_eq!(d.kind, EventKind::Alloc);
        assert_eq!(d.addr, 0xdead_beef);
        assert_eq!(d.size, 4096);
        assert_eq!(d.ts_nanos, 123_456_789);
        assert_eq!(d.site, SiteId(37));
        assert_eq!(d.thread, 7);
        assert_eq!(d.align, 16); // align_log2 round-trips back to 16
    }

    #[test]
    fn site_and_batch_stream() {
        let mut b = Vec::new();
        encode_header(&mut b, 1, "x");
        let frames = [FrameRec {
            function: "app::main",
            file: "main.rs",
            line: 42,
            inlined: false,
        }];
        encode_site(&mut b, 5, Some("app::Widget"), Some(AllocShape::Boxed), &frames);
        encode_events_header(&mut b, 2);
        for i in 0..2u64 {
            encode_event(
                &mut b,
                &RawEvent {
                    kind: EventKind::Alloc,
                    seq: 0,
                    ts_nanos: i,
                    addr: 0x1000 + i,
                    size: 64,
                    align: 8,
                    site: SiteId(5),
                    thread: 1,
                },
            );
        }
        // Decode: header, then a site tag, then an events tag with 2 records.
        let (_h, used) = decode_header(&b).unwrap();
        let mut r = Reader::new(&b);
        r.pos = used;
        assert_eq!(r.u8(), Some(TAG_SITE));
        assert_eq!(r.u32(), Some(5));
        assert_eq!(r.u8(), Some(1)); // has_ty
        assert_eq!(r.str().as_deref(), Some("app::Widget"));
        assert_eq!(shape_from_code(r.u8().unwrap()), Some(AllocShape::Boxed));
        assert_eq!(r.u16(), Some(1)); // nframes
        assert_eq!(r.str().as_deref(), Some("app::main"));
        assert_eq!(r.str().as_deref(), Some("main.rs"));
        assert_eq!(r.u32(), Some(42));
        assert_eq!(r.u8(), Some(0)); // inlined
        assert_eq!(r.u8(), Some(TAG_EVENTS));
        assert_eq!(r.u32(), Some(2));
        let e0 = r.decode_event().unwrap();
        assert_eq!(e0.addr, 0x1000);
        let e1 = r.decode_event().unwrap();
        assert_eq!(e1.addr, 0x1001);
    }
}
