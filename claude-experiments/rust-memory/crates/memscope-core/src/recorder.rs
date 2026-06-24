//! The recording engine: the process-global state behind the tracking
//! allocator. Holds the live-allocation table, the allocation-site interner,
//! the event ring buffer, runtime configuration, and aggregate stats.
//!
//! Everything here is touched *inside* the allocator reentrancy guard (see
//! `lib.rs`), so the bookkeeping is free to use ordinary `std` collections:
//! their own allocations bypass recording rather than recursing.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use memscope_proto::{EventKind, Frame, LiveAlloc, RawEvent, SiteId, SiteInfo, Snapshot, TypeId};

/// Max stack depth captured per allocation site.
pub const MAX_FRAMES: usize = 64;
/// Number of live-table shards (power of two).
const SHARDS: usize = 64;
/// Default event-ring capacity (events; older ones drop on overflow).
const DEFAULT_RING_CAP: usize = 1 << 16;

/// Recording mode. Switchable at runtime via [`Recorder::set_mode`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Mode {
    /// Pass through to the inner allocator; record nothing.
    Off = 0,
    /// Record every allocation: exact live set, exact heap dumps.
    Full = 1,
    /// Record ~1/N allocations; aggregates scale by N. Low overhead.
    Sampled = 2,
}

impl Mode {
    fn from_u8(v: u8) -> Mode {
        match v {
            1 => Mode::Full,
            2 => Mode::Sampled,
            _ => Mode::Off,
        }
    }
}

/// Per-live-allocation bookkeeping. Packed to 24 bytes (from 32): the timestamp
/// is coarsened to milliseconds and the alignment to its log2, since neither
/// needs full precision in the live table (events carry exact values).
struct Entry {
    size: u64,
    ts_ms: u32,
    site: SiteId,
    thread: u32,
    align_log2: u8,
}

pub struct Recorder {
    start: Instant,

    // --- runtime config ---
    mode: AtomicU8,
    sample_rate: AtomicU32, // record 1 in N (Sampled mode)
    bt_depth: AtomicUsize,
    capture_sites: std::sync::atomic::AtomicBool,

    // --- live allocation table, sharded by address ---
    shards: [Mutex<HashMap<u64, Entry>>; SHARDS],

    // --- allocation-site interner ---
    sites: Mutex<SiteInterner>,

    // --- event ring ---
    ring: Mutex<VecDeque<RawEvent>>,
    ring_cap: usize,

    // --- aggregate stats ---
    seq: AtomicU64,
    sample_counter: AtomicU64,
    live_bytes: AtomicU64,
    total_allocs: AtomicU64,
    total_alloc_bytes: AtomicU64,
    dropped_events: AtomicU64,
}

/// Interns captured stack traces (slices of return addresses) into small
/// [`SiteId`]s. Bucketed by a cheap FNV hash to avoid allocating a lookup key
/// on every allocation; only genuinely new sites allocate.
struct SiteInterner {
    buckets: HashMap<u64, Vec<u32>>, // hash -> candidate site ids
    frames: Vec<Box<[usize]>>,       // site id -> return addresses
}

impl SiteInterner {
    fn new() -> Self {
        SiteInterner {
            buckets: HashMap::new(),
            frames: Vec::new(),
        }
    }

    fn intern(&mut self, frames: &[usize]) -> SiteId {
        let h = hash_frames(frames);
        let bucket = self.buckets.entry(h).or_default();
        for &id in bucket.iter() {
            if self.frames[id as usize].as_ref() == frames {
                return SiteId(id);
            }
        }
        let id = self.frames.len() as u32;
        self.frames.push(frames.to_vec().into_boxed_slice());
        bucket.push(id);
        SiteId(id)
    }
}

/// Hash a captured stack trace. Mixes one *word* per frame (an FNV-style
/// multiply) instead of one byte, so a 64-frame trace costs ~64 multiplies
/// rather than ~512 byte-steps — material on the (now backtrace-free) hot path.
#[inline]
fn hash_frames(frames: &[usize]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &f in frames {
        h = (h ^ f as u64).wrapping_mul(0x100000001b3);
    }
    h
}

#[inline]
fn shard_for(addr: u64) -> usize {
    // Spread out the (often 16-byte-aligned) low bits.
    ((addr >> 6) ^ (addr >> 18)) as usize & (SHARDS - 1)
}

impl Recorder {
    fn new() -> Self {
        Recorder {
            start: Instant::now(),
            mode: AtomicU8::new(Mode::Off as u8),
            sample_rate: AtomicU32::new(1),
            bt_depth: AtomicUsize::new(MAX_FRAMES),
            capture_sites: std::sync::atomic::AtomicBool::new(true),
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            sites: Mutex::new(SiteInterner::new()),
            ring: Mutex::new(VecDeque::with_capacity(1024)),
            ring_cap: DEFAULT_RING_CAP,
            seq: AtomicU64::new(0),
            sample_counter: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
            total_allocs: AtomicU64::new(0),
            total_alloc_bytes: AtomicU64::new(0),
            dropped_events: AtomicU64::new(0),
        }
    }

    #[inline]
    fn now_nanos(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    #[inline]
    pub fn mode(&self) -> Mode {
        Mode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    pub fn set_mode(&self, mode: Mode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
    }

    pub fn set_sample_rate(&self, rate: u32) {
        self.sample_rate.store(rate.max(1), Ordering::Relaxed);
    }

    pub fn set_backtrace_depth(&self, depth: usize) {
        self.bt_depth
            .store(depth.clamp(1, MAX_FRAMES), Ordering::Relaxed);
    }

    pub fn set_capture_sites(&self, on: bool) {
        self.capture_sites.store(on, Ordering::Relaxed);
    }

    /// Should this allocation be recorded? Returns the per-site sample scale (1
    /// for Full; N for Sampled) when yes, or `None` to skip.
    #[inline]
    fn should_record(&self) -> Option<f64> {
        match self.mode() {
            Mode::Off => None,
            Mode::Full => Some(1.0),
            Mode::Sampled => {
                let n = self.sample_rate.load(Ordering::Relaxed).max(1) as u64;
                let c = self.sample_counter.fetch_add(1, Ordering::Relaxed);
                if c % n == 0 {
                    Some(n as f64)
                } else {
                    None
                }
            }
        }
    }

    /// Record an allocation. `unwind` is invoked to capture the site (already
    /// inside the reentrancy guard). Returns the event we pushed, if recorded.
    pub fn on_alloc(
        &self,
        addr: u64,
        size: u64,
        align: u32,
        capture: impl FnOnce(&mut [usize], usize) -> usize,
        kind: EventKind,
    ) {
        if self.should_record().is_none() {
            return;
        }
        let thread = thread_id();
        let ts = self.now_nanos();

        let site = if self.capture_sites.load(Ordering::Relaxed) {
            let mut buf = [0usize; MAX_FRAMES];
            let depth = self.bt_depth.load(Ordering::Relaxed);
            let n = capture(&mut buf[..depth], 0);
            if n > 0 {
                self.sites.lock().unwrap().intern(&buf[..n])
            } else {
                SiteId::NONE
            }
        } else {
            SiteId::NONE
        };

        let seq = self.seq.fetch_add(1, Ordering::Relaxed);

        {
            let sh = &self.shards[shard_for(addr)];
            sh.lock().unwrap().insert(
                addr,
                Entry {
                    size,
                    ts_ms: (ts / 1_000_000) as u32,
                    site,
                    thread,
                    align_log2: align.max(1).trailing_zeros() as u8,
                },
            );
        }

        self.live_bytes.fetch_add(size, Ordering::Relaxed);
        self.total_allocs.fetch_add(1, Ordering::Relaxed);
        self.total_alloc_bytes.fetch_add(size, Ordering::Relaxed);

        self.push_event(RawEvent {
            kind,
            seq,
            ts_nanos: ts,
            addr,
            size,
            align,
            site,
            thread,
        });
    }

    /// Record a deallocation. No-op if the address was never tracked (e.g. it
    /// wasn't sampled). Returns nothing; emits a Dealloc event if it was live.
    pub fn on_dealloc(&self, addr: u64) {
        if self.mode() == Mode::Off {
            return;
        }
        let entry = {
            let sh = &self.shards[shard_for(addr)];
            sh.lock().unwrap().remove(&addr)
        };
        let Some(entry) = entry else { return };

        self.live_bytes
            .fetch_sub(entry.size.min(self.live_bytes.load(Ordering::Relaxed)), Ordering::Relaxed);

        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let ts = self.now_nanos();
        self.push_event(RawEvent {
            kind: EventKind::Dealloc,
            seq,
            ts_nanos: ts,
            addr,
            size: entry.size,
            align: 1u32 << entry.align_log2,
            site: entry.site,
            thread: entry.thread,
        });
    }

    #[inline]
    fn push_event(&self, ev: RawEvent) {
        let mut ring = self.ring.lock().unwrap();
        if ring.len() >= self.ring_cap {
            ring.pop_front();
            self.dropped_events.fetch_add(1, Ordering::Relaxed);
        }
        ring.push_back(ev);
    }

    /// Drain up to `max` queued events into `out` (oldest first). Returns the
    /// number drained. Consumers poll this to follow the live stream.
    pub fn drain_events(&self, out: &mut Vec<RawEvent>, max: usize) -> usize {
        let mut ring = self.ring.lock().unwrap();
        let take = max.min(ring.len());
        out.extend(ring.drain(..take));
        take
    }

    pub fn stats(&self) -> Stats {
        Stats {
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            total_allocs: self.total_allocs.load(Ordering::Relaxed),
            total_alloc_bytes: self.total_alloc_bytes.load(Ordering::Relaxed),
            dropped_events: self.dropped_events.load(Ordering::Relaxed),
            mode: self.mode(),
            sample_rate: self.sample_rate.load(Ordering::Relaxed),
        }
    }

    /// Build a self-contained heap dump of the current live set. Site frames are
    /// captured as raw return addresses; symbolication + type recovery is the
    /// `memscope-symbols` crate's job (it fills `function`/`ty`).
    pub fn snapshot(&self) -> Snapshot {
        let scale = match self.mode() {
            Mode::Sampled => self.sample_rate.load(Ordering::Relaxed).max(1) as f64,
            _ => 1.0,
        };

        let mut live = Vec::new();
        let mut used_sites: HashMap<u32, ()> = HashMap::new();
        for sh in &self.shards {
            let g = sh.lock().unwrap();
            for (&addr, e) in g.iter() {
                if e.site.is_some() {
                    used_sites.insert(e.site.0, ());
                }
                live.push(LiveAlloc {
                    addr,
                    size: e.size,
                    align: 1u32 << e.align_log2,
                    site: e.site,
                    ts_nanos: e.ts_ms as u64 * 1_000_000,
                    thread: e.thread,
                });
            }
        }

        let total_live_bytes: u64 = live.iter().map(|l| l.size).sum();

        let sites = {
            let interner = self.sites.lock().unwrap();
            used_sites
                .keys()
                .map(|&id| SiteInfo {
                    id,
                    frames: interner.frames[id as usize]
                        .iter()
                        .map(|&ip| Frame {
                            ip: ip as u64,
                            ..Default::default()
                        })
                        .collect(),
                    ty: TypeId::UNKNOWN,
                    shape: None,
                })
                .collect()
        };

        Snapshot {
            taken_at_nanos: self.now_nanos(),
            sample_scale: scale,
            live,
            sites,
            types: Vec::new(),
            total_live_bytes,
            dropped_events: self.dropped_events.load(Ordering::Relaxed),
        }
    }

    /// Raw return-address frames for an interned site (for external
    /// symbolication without a full snapshot).
    pub fn site_frames(&self, site: SiteId) -> Option<Vec<u64>> {
        if !site.is_some() {
            return None;
        }
        let interner = self.sites.lock().unwrap();
        interner
            .frames
            .get(site.0 as usize)
            .map(|f| f.iter().map(|&ip| ip as u64).collect())
    }
}

/// Aggregate counters, cheap to read at any time.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub live_bytes: u64,
    pub total_allocs: u64,
    pub total_alloc_bytes: u64,
    pub dropped_events: u64,
    pub mode: Mode,
    pub sample_rate: u32,
}

static RECORDER: OnceLock<Recorder> = OnceLock::new();

/// Access the process-global recorder. Callers on the allocation hot path must
/// already hold the reentrancy guard, because first call here allocates.
#[inline]
pub fn recorder() -> &'static Recorder {
    RECORDER.get_or_init(Recorder::new)
}

// --- cheap numeric thread ids -------------------------------------------------

static NEXT_THREAD_ID: AtomicU32 = AtomicU32::new(1);

thread_local! {
    static THREAD_ID: u32 = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
}

#[inline]
fn thread_id() -> u32 {
    THREAD_ID.with(|&id| id)
}
