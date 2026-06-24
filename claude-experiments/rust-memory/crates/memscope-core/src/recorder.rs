//! The recording engine: the process-global state behind the tracking
//! allocator. Holds the live-allocation table, the allocation-site interner,
//! the event ring buffer, runtime configuration, and aggregate stats.
//!
//! Everything here is touched *inside* the allocator reentrancy guard (see
//! `lib.rs`), so the bookkeeping is free to use ordinary `std` collections:
//! their own allocations bypass recording rather than recursing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use memscope_proto::{EventKind, Frame, LiveAlloc, RawEvent, SiteId, SiteInfo, Snapshot, TypeId};

/// Max stack depth captured per allocation site.
pub const MAX_FRAMES: usize = 64;
/// Number of live-table shards (power of two). High enough that concurrent
/// threads rarely collide on the same shard lock.
const SHARDS: usize = 64;
/// Number of site-interner shards (power of two); ids embed the shard index.
const SITE_SHARDS: usize = 64;
const SITE_SHARD_BITS: u32 = 6; // log2(SITE_SHARDS)
const SITE_SHARD_MASK: u32 = (SITE_SHARDS as u32) - 1;
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
    /// Event streaming is opt-in: the ring (a contended global mutex) is only
    /// written when a consumer is actually draining events. Off by default, so
    /// the common snapshot/graph workflow never pays for it.
    events_enabled: std::sync::atomic::AtomicBool,

    // --- live allocation table, sharded by address ---
    shards: [Mutex<Shard>; SHARDS],

    // --- allocation-site interner, sharded to avoid a global lock ---
    site_shards: [Mutex<SiteShard>; SITE_SHARDS],

    // --- event ring (only written when events_enabled) ---
    ring: crate::ring::Ring,

    // --- aggregate stats ---
    seq: AtomicU64,
    sample_counter: AtomicU64,
}

/// One shard of the live table, with its own stats counters. Keeping the
/// counters *inside* the shard means they're bumped under the lock we already
/// hold — no globally-contended atomics on the hot path (the bottleneck that
/// otherwise serializes every thread on the same few cache lines).
struct Shard {
    map: HashMap<u64, Entry>,
    live_bytes: u64,
    total_allocs: u64,
    total_alloc_bytes: u64,
}

impl Shard {
    fn new() -> Self {
        Shard {
            map: HashMap::new(),
            live_bytes: 0,
            total_allocs: 0,
            total_alloc_bytes: 0,
        }
    }
}

/// One shard of the allocation-site interner. Site ids embed their shard in the
/// low [`SITE_SHARD_BITS`] bits, so a lookup goes straight to the owning shard
/// without a global lock.
struct SiteShard {
    buckets: HashMap<u64, Vec<u32>>,    // frame-hash -> site ids in this shard
    frames: HashMap<u32, Box<[usize]>>, // site id -> return addresses
    next_local: u32,
}

impl SiteShard {
    fn new() -> Self {
        SiteShard {
            buckets: HashMap::new(),
            frames: HashMap::new(),
            next_local: 0,
        }
    }

    fn intern(&mut self, shard_idx: usize, h: u64, frames: &[usize]) -> SiteId {
        let existing = self
            .buckets
            .get(&h)
            .and_then(|ids| ids.iter().find(|&&id| self.frames[&id].as_ref() == frames).copied());
        if let Some(id) = existing {
            return SiteId(id);
        }
        let local = self.next_local;
        self.next_local += 1;
        let id = (local << SITE_SHARD_BITS) | shard_idx as u32;
        self.frames.insert(id, frames.to_vec().into_boxed_slice());
        self.buckets.entry(h).or_default().push(id);
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
            events_enabled: std::sync::atomic::AtomicBool::new(false),
            shards: std::array::from_fn(|_| Mutex::new(Shard::new())),
            site_shards: std::array::from_fn(|_| Mutex::new(SiteShard::new())),
            ring: crate::ring::Ring::new(DEFAULT_RING_CAP, crate::ring::RingMode::Overwrite),
            seq: AtomicU64::new(0),
            sample_counter: AtomicU64::new(0),
        }
    }

    #[inline]
    fn now_nanos(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Coarse milliseconds-since-start for the live table, amortizing the clock
    /// read across many allocations per thread (the table only keeps ms anyway).
    #[inline]
    fn now_ms(&self) -> u32 {
        CLOCK_CACHE.with(|c| {
            let (ms, countdown) = c.get();
            if countdown == 0 {
                let now = self.start.elapsed().as_millis() as u32;
                c.set((now, 255));
                now
            } else {
                c.set((ms, countdown - 1));
                ms
            }
        })
    }

    pub fn set_event_streaming(&self, on: bool) {
        self.events_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Intern a captured stack trace into a [`SiteId`]. A per-thread cache of the
    /// most-recent sites makes the common case — a tight loop allocating at the
    /// same few sites — completely lock-free, so a hot site doesn't serialize all
    /// threads on one interner shard.
    #[inline]
    fn intern_site(&self, frames: &[usize]) -> SiteId {
        let h = hash_frames(frames);
        if let Some(id) = SITE_CACHE.with(|c| c.borrow().lookup(h, frames)) {
            return id;
        }
        let s = (h as usize) & (SITE_SHARDS - 1);
        let id = self.site_shards[s].lock().unwrap().intern(s, h, frames);
        SITE_CACHE.with(|c| c.borrow_mut().insert(h, frames, id));
        id
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
        let ts_ms = self.now_ms();

        let site = if self.capture_sites.load(Ordering::Relaxed) {
            let mut buf = [0usize; MAX_FRAMES];
            let depth = self.bt_depth.load(Ordering::Relaxed);
            let n = capture(&mut buf[..depth], 0);
            if n > 0 {
                self.intern_site(&buf[..n])
            } else {
                SiteId::NONE
            }
        } else {
            SiteId::NONE
        };

        {
            let mut shard = self.shards[shard_for(addr)].lock().unwrap();
            shard.map.insert(
                addr,
                Entry {
                    size,
                    ts_ms,
                    site,
                    thread,
                    align_log2: align.max(1).trailing_zeros() as u8,
                },
            );
            // Stats updated under the shard lock we already hold — no globally
            // contended atomics.
            shard.live_bytes += size;
            shard.total_allocs += 1;
            shard.total_alloc_bytes += size;
        }

        // Event streaming is opt-in: skip the contended ring (and the sequence
        // counter + precise clock read) unless a consumer is draining.
        if self.events_enabled.load(Ordering::Relaxed) {
            let seq = self.seq.fetch_add(1, Ordering::Relaxed);
            self.push_event(RawEvent {
                kind,
                seq,
                ts_nanos: self.now_nanos(),
                addr,
                size,
                align,
                site,
                thread,
            });
        }
    }

    /// Record a deallocation. No-op if the address was never tracked (e.g. it
    /// wasn't sampled). Returns nothing; emits a Dealloc event if it was live.
    pub fn on_dealloc(&self, addr: u64) {
        if self.mode() == Mode::Off {
            return;
        }
        let entry = {
            let mut shard = self.shards[shard_for(addr)].lock().unwrap();
            let entry = shard.map.remove(&addr);
            if let Some(e) = &entry {
                shard.live_bytes = shard.live_bytes.saturating_sub(e.size);
            }
            entry
        };
        let Some(entry) = entry else { return };

        if !self.events_enabled.load(Ordering::Relaxed) {
            return;
        }
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
        self.ring.push(ev);
    }

    /// Switch the event ring between overwrite and reliable (backpressure) modes.
    pub fn set_ring_mode(&self, mode: crate::ring::RingMode) {
        self.ring.set_mode(mode);
    }

    /// Total events dropped by the ring (consumer fell behind).
    pub fn ring_dropped(&self) -> u64 {
        self.ring.dropped()
    }

    /// Drain all currently-available events into `out` (sequence order). Must be
    /// called by a single consumer (the pump, or `drain_events`). Returns the
    /// number drained.
    pub fn drain_ring(&self, out: &mut Vec<RawEvent>) -> usize {
        self.ring.drain(out)
    }

    /// Backwards-compatible drain used by the agent's PollEvents. `max` caps the
    /// returned batch; any beyond it stay for the next call.
    pub fn drain_events(&self, out: &mut Vec<RawEvent>, max: usize) -> usize {
        let before = out.len();
        self.ring.drain(out);
        if out.len() - before > max {
            out.truncate(before + max);
        }
        out.len() - before
    }

    pub fn stats(&self) -> Stats {
        let (mut live_bytes, mut total_allocs, mut total_alloc_bytes) = (0u64, 0u64, 0u64);
        for sh in &self.shards {
            let s = sh.lock().unwrap();
            live_bytes += s.live_bytes;
            total_allocs += s.total_allocs;
            total_alloc_bytes += s.total_alloc_bytes;
        }
        Stats {
            live_bytes,
            total_allocs,
            total_alloc_bytes,
            dropped_events: self.ring.dropped(),
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
            for (&addr, e) in g.map.iter() {
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

        let sites = used_sites
            .keys()
            .map(|&id| SiteInfo {
                id,
                frames: self
                    .site_frames(SiteId(id))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|ip| Frame {
                        ip,
                        ..Default::default()
                    })
                    .collect(),
                ty: TypeId::UNKNOWN,
                shape: None,
            })
            .collect();

        Snapshot {
            taken_at_nanos: self.now_nanos(),
            sample_scale: scale,
            live,
            sites,
            types: Vec::new(),
            total_live_bytes,
            dropped_events: self.ring.dropped(),
        }
    }

    /// Raw return-address frames for an interned site (for external
    /// symbolication without a full snapshot).
    pub fn site_frames(&self, site: SiteId) -> Option<Vec<u64>> {
        if !site.is_some() {
            return None;
        }
        let s = (site.0 & SITE_SHARD_MASK) as usize;
        let shard = self.site_shards[s].lock().unwrap();
        shard
            .frames
            .get(&site.0)
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
    /// Per-thread coarse clock cache: (cached_ms, countdown). Refreshes the
    /// millisecond reading every 256 allocations to amortize the clock syscall.
    static CLOCK_CACHE: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    /// Per-thread recently-interned sites, so a hot site needs no shared lock.
    static SITE_CACHE: std::cell::RefCell<SiteCache> =
        const { std::cell::RefCell::new(SiteCache::new()) };
}

/// A tiny per-thread LRU of recently-interned sites. Holds a few entries so a
/// loop allocating at 2–3 sites (e.g. a `Vec` and a `String`) stays lock-free.
const SITE_CACHE_SLOTS: usize = 4;

struct SiteCache {
    hashes: [u64; SITE_CACHE_SLOTS],
    frames: [Option<Box<[usize]>>; SITE_CACHE_SLOTS],
    ids: [SiteId; SITE_CACHE_SLOTS],
    next: usize,
}

impl SiteCache {
    const fn new() -> Self {
        SiteCache {
            hashes: [0; SITE_CACHE_SLOTS],
            frames: [None, None, None, None],
            ids: [SiteId::NONE; SITE_CACHE_SLOTS],
            next: 0,
        }
    }

    #[inline]
    fn lookup(&self, h: u64, frames: &[usize]) -> Option<SiteId> {
        for i in 0..SITE_CACHE_SLOTS {
            if self.hashes[i] == h {
                if let Some(f) = &self.frames[i] {
                    if f.as_ref() == frames {
                        return Some(self.ids[i]);
                    }
                }
            }
        }
        None
    }

    #[inline]
    fn insert(&mut self, h: u64, frames: &[usize], id: SiteId) {
        let i = self.next;
        self.next = (self.next + 1) % SITE_CACHE_SLOTS;
        self.hashes[i] = h;
        self.frames[i] = Some(frames.to_vec().into_boxed_slice());
        self.ids[i] = id;
    }
}

#[inline]
fn thread_id() -> u32 {
    THREAD_ID.with(|&id| id)
}
