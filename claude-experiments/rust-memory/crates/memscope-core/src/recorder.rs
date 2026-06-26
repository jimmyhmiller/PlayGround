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

use memscope_proto::{
    EventKind, Frame, LiveAlloc, MetaValue, RawEvent, SiteId, SiteInfo, Snapshot, TypeId,
};

/// Max stack depth captured per allocation site.
pub const MAX_FRAMES: usize = 64;
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

pub struct Recorder {
    start: Instant,

    // --- runtime config ---
    mode: AtomicU8,
    sample_rate: AtomicU32, // record 1 in N (Sampled mode)
    bt_depth: AtomicUsize,
    capture_sites: std::sync::atomic::AtomicBool,

    // --- allocation-site interner, sharded to avoid a global lock ---
    site_shards: [Mutex<SiteShard>; SITE_SHARDS],

    // --- metadata interners (the `meta!` path; coarse, so a plain lock is fine) ---
    keys: Mutex<KeyInterner>,
    meta: Mutex<MetaInterner>,

    // --- the hot-path event sink: per-thread sharded rings ---
    ring: crate::tls_ring::ShardedRing,

    // --- consumer side (off the hot path), fed by the single pump thread ---
    /// The in-process reconstructed live set; backs `snapshot()` / `stats()`.
    reconstruct: Mutex<crate::sink::LiveSet>,
    /// Recent raw events buffered for the agent's PollEvents stream.
    recent: Mutex<VecDeque<RawEvent>>,
    /// Extra user-registered sinks (file / socket / custom) fed by the pump.
    user_sinks: Mutex<Vec<Box<dyn crate::sink::EventSink>>>,
    /// Highest event timestamp the pump has *applied* to `reconstruct` (snapshot
    /// waits on this so it reflects all events up to call time).
    applied_tsc: AtomicU64,
    /// Calibrated timestamp frequency (ticks/sec), set by the pump on start.
    tsc_hz: AtomicU64,
    pump_started: std::sync::Once,

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

impl Recorder {
    fn new() -> Self {
        Recorder {
            start: Instant::now(),
            mode: AtomicU8::new(Mode::Off as u8),
            sample_rate: AtomicU32::new(1),
            bt_depth: AtomicUsize::new(MAX_FRAMES),
            capture_sites: std::sync::atomic::AtomicBool::new(true),
            site_shards: std::array::from_fn(|_| Mutex::new(SiteShard::new())),
            keys: Mutex::new(KeyInterner::default()),
            meta: Mutex::new(MetaInterner::default()),
            ring: crate::tls_ring::ShardedRing::new(
                DEFAULT_RING_CAP,
                crate::tls_ring::RingMode::Overwrite,
            ),
            reconstruct: Mutex::new(crate::sink::LiveSet::new()),
            recent: Mutex::new(VecDeque::new()),
            user_sinks: Mutex::new(Vec::new()),
            applied_tsc: AtomicU64::new(0),
            tsc_hz: AtomicU64::new(0),
            pump_started: std::sync::Once::new(),
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
        CLOCK_CACHE
            .try_with(|c| {
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
            .unwrap_or(0)
    }

    /// Kept for API compatibility: the event ring is now the core mechanism, so
    /// streaming is always on while recording. Turning it "on" just ensures the
    /// reconstructor pump is running.
    pub fn set_event_streaming(&self, on: bool) {
        if on {
            self.ensure_pump();
        }
    }

    /// Intern a captured stack trace into a [`SiteId`]. A per-thread cache of the
    /// most-recent sites makes the common case — a tight loop allocating at the
    /// same few sites — completely lock-free, so a hot site doesn't serialize all
    /// threads on one interner shard.
    #[inline]
    fn intern_site(&self, frames: &[usize]) -> SiteId {
        let h = hash_frames(frames);
        if let Some(id) = SITE_CACHE
            .try_with(|c| c.borrow().lookup(h, frames))
            .ok()
            .flatten()
        {
            return id;
        }
        let s = (h as usize) & (SITE_SHARDS - 1);
        let id = self.site_shards[s].lock().unwrap().intern(s, h, frames);
        let _ = SITE_CACHE.try_with(|c| c.borrow_mut().insert(h, frames, id));
        id
    }

    #[inline]
    pub fn mode(&self) -> Mode {
        Mode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    pub fn set_mode(&self, mode: Mode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
        // Tracking implies the reconstructor must run from now on, so it sees
        // every event (long-lived allocations included) — not just whatever is
        // still in the ring when the first snapshot is taken.
        if mode != Mode::Off && std::env::var_os("MEMSCOPE_NOPUMP").is_none() {
            self.ensure_pump();
        }
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
                // Per-thread counter — no globally-contended atomic. Each thread
                // independently samples 1/n of its own allocations.
                let c = SAMPLE_COUNTER
                    .try_with(|c| {
                        let v = c.get();
                        c.set(v.wrapping_add(1));
                        v
                    })
                    .unwrap_or(0);
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

        // The whole hot path: append a flat record to this thread's own ring.
        // `seq` is a hardware timestamp — the global causal order key the
        // consumer merges on (see tls_ring).
        self.ring.push(RawEvent {
            kind,
            seq: crate::tls_ring::tsc(),
            ts_nanos: ts_ms as u64 * 1_000_000,
            addr,
            size,
            align,
            site,
            thread,
        });
    }

    /// Record a deallocation. The size/align come straight from the caller's
    /// `Layout`, so no table lookup is needed — just append a free record.
    pub fn on_dealloc(&self, addr: u64, size: u64, align: u32) {
        if self.mode() == Mode::Off {
            return;
        }
        self.ring.push(RawEvent {
            kind: EventKind::Dealloc,
            seq: crate::tls_ring::tsc(),
            ts_nanos: self.now_ms() as u64 * 1_000_000,
            addr,
            size,
            align,
            site: SiteId::NONE,
            thread: thread_id(),
        });
    }

    /// The single consumer: drain every per-thread ring, merge by timestamp
    /// behind a small watermark (so a lagging ring can't deliver an out-of-order
    /// event after we've applied past it), and apply to the reconstructor, the
    /// recent-events buffer, and any user sinks. Started once, lazily.
    fn ensure_pump(&self) {
        self.pump_started.call_once(|| {
            std::thread::Builder::new()
                .name("memscope-reconstruct".into())
                .spawn(|| {
                    crate::exclude_current_thread();
                    use crate::sink::EventSink;
                    let r = recorder();
                    let hz = crate::tls_ring::calibrate_tsc_hz();
                    r.tsc_hz.store(hz, Ordering::Release);
                    // Reorder window: events newer than this (in ticks) are held
                    // until peers can't undercut them. ~1ms.
                    let window = (hz / 1000).max(1);

                    let mut pending: Vec<RawEvent> = Vec::with_capacity(16384);
                    loop {
                        let drained = r.ring.drain_all(&mut pending);
                        if drained == 0 && pending.is_empty() {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                            continue;
                        }
                        // Stable order by timestamp.
                        pending.sort_by_key(|e| e.seq);
                        let watermark = crate::tls_ring::tsc().saturating_sub(window);
                        // Apply the prefix older than the watermark.
                        let split = pending.partition_point(|e| e.seq <= watermark);
                        if split > 0 {
                            let batch = &pending[..split];
                            r.reconstruct.lock().unwrap().consume(batch);
                            {
                                let mut recent = r.recent.lock().unwrap();
                                for e in batch {
                                    if recent.len() >= DEFAULT_RING_CAP {
                                        recent.pop_front();
                                    }
                                    recent.push_back(*e);
                                }
                            }
                            {
                                let mut sinks = r.user_sinks.lock().unwrap();
                                for s in sinks.iter_mut() {
                                    s.consume(batch);
                                }
                            }
                            pending.drain(..split);
                        }
                        // Everything up to the watermark is now applied.
                        r.applied_tsc.store(watermark, Ordering::Release);
                        if split == 0 {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                        }
                    }
                })
                .expect("failed to spawn memscope reconstructor thread");
        });
    }

    /// Wait until the reconstructor has applied every event up to this instant
    /// (bounded, so a dead pump can't hang the caller).
    fn flush(&self) {
        let target = crate::tls_ring::tsc();
        let start = Instant::now();
        while self.applied_tsc.load(Ordering::Acquire) < target {
            if start.elapsed() > std::time::Duration::from_millis(500) {
                break;
            }
            std::thread::yield_now();
        }
    }

    /// Register an extra sink (file / socket / custom). Fed by the pump.
    pub fn add_sink(&self, sink: Box<dyn crate::sink::EventSink>) {
        self.ensure_pump();
        self.user_sinks.lock().unwrap().push(sink);
    }

    /// Switch the event ring between overwrite and reliable (backpressure) modes.
    pub fn set_ring_mode(&self, mode: crate::tls_ring::RingMode) {
        self.ring.set_mode(mode);
    }

    /// Total events dropped by the ring (consumer fell behind).
    pub fn ring_dropped(&self) -> u64 {
        self.ring.dropped()
    }

    /// Pop up to `max` buffered events (oldest first) for the agent's live
    /// stream. These come from the pump's recent-events buffer, not the ring
    /// directly (the pump is the ring's single consumer).
    pub fn drain_events(&self, out: &mut Vec<RawEvent>, max: usize) -> usize {
        self.ensure_pump();
        let mut recent = self.recent.lock().unwrap();
        let take = max.min(recent.len());
        out.extend(recent.drain(..take));
        take
    }

    pub fn stats(&self) -> Stats {
        self.ensure_pump();
        self.flush();
        let rec = self.reconstruct.lock().unwrap();
        Stats {
            live_bytes: rec.live_bytes(),
            total_allocs: rec.total_allocs(),
            total_alloc_bytes: rec.total_alloc_bytes(),
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

        self.ensure_pump();
        self.flush();

        let mut live = Vec::new();
        let mut used_sites: HashMap<u32, ()> = HashMap::new();
        let total_live_bytes;
        {
            let rec = self.reconstruct.lock().unwrap();
            live.reserve(rec.live_count());
            for (addr, e) in rec.iter() {
                if e.site.is_some() {
                    used_sites.insert(e.site.0, ());
                }
                live.push(LiveAlloc {
                    addr,
                    size: e.size,
                    align: e.align,
                    site: e.site,
                    ts_nanos: e.ts_nanos,
                    thread: e.thread,
                });
            }
            total_live_bytes = rec.live_bytes();
        }

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

    // --- metadata (`meta!`) -------------------------------------------------

    /// Intern a metadata key name to a stable id.
    pub fn intern_key(&self, name: &str) -> u32 {
        self.keys.lock().unwrap().intern(name)
    }

    /// Intern a metadata context (a scope's key/value pairs) to a stable id.
    pub fn intern_meta(&self, kvs: &[(u32, MetaValue)]) -> u32 {
        self.meta.lock().unwrap().intern(kvs)
    }

    /// Record entering/leaving a metadata scope: a marker event carrying the
    /// context id in the `site` slot. Does nothing when recording is off.
    pub fn on_meta(&self, kind: EventKind, meta_id: u32) {
        if self.mode() == Mode::Off {
            return;
        }
        self.ring.push(RawEvent {
            kind,
            seq: crate::tls_ring::tsc(),
            ts_nanos: self.now_ms() as u64 * 1_000_000,
            addr: 0,
            size: 0,
            align: 0,
            site: SiteId(meta_id),
            thread: thread_id(),
        });
    }

    /// Key name for an interned key id (for the file recorder's `TAG_KEY` table).
    pub fn key_name(&self, key_id: u32) -> Option<String> {
        self.keys.lock().unwrap().names.get(key_id as usize).cloned()
    }

    /// The key/value pairs of an interned context (for the `TAG_META` table).
    pub fn meta_context(&self, meta_id: u32) -> Option<Vec<(u32, MetaValue)>> {
        self.meta.lock().unwrap().contexts.get(&meta_id).cloned()
    }
}

/// Interns metadata key names. Tiny — a program has a handful of distinct keys.
#[derive(Default)]
struct KeyInterner {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl KeyInterner {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }
}

/// Interns metadata contexts (key/value sets) to ids, deduped by content.
#[derive(Default)]
struct MetaInterner {
    buckets: HashMap<u64, Vec<u32>>,
    contexts: HashMap<u32, Vec<(u32, MetaValue)>>,
    next: u32,
}

impl MetaInterner {
    fn intern(&mut self, kvs: &[(u32, MetaValue)]) -> u32 {
        let h = hash_kvs(kvs);
        if let Some(ids) = self.buckets.get(&h) {
            for &id in ids {
                if self.contexts.get(&id).map(|c| c.as_slice() == kvs).unwrap_or(false) {
                    return id;
                }
            }
        }
        let id = self.next;
        self.next += 1;
        self.contexts.insert(id, kvs.to_vec());
        self.buckets.entry(h).or_default().push(id);
        id
    }
}

fn hash_kvs(kvs: &[(u32, MetaValue)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (k, v) in kvs {
        h = (h ^ *k as u64).wrapping_mul(0x100000001b3);
        let vh = match v {
            MetaValue::Str(s) => {
                let mut x: u64 = 0;
                for b in s.bytes() {
                    x = (x ^ b as u64).wrapping_mul(0x100000001b3);
                }
                x
            }
            MetaValue::Int(i) => *i as u64,
            MetaValue::Uint(u) => *u,
            MetaValue::F64(f) => f.to_bits(),
            MetaValue::Bool(b) => *b as u64,
        };
        h = (h ^ vh).wrapping_mul(0x100000001b3);
    }
    h
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
    /// Per-thread sampling counter (avoids a globally-contended atomic).
    static SAMPLE_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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
    THREAD_ID.try_with(|&id| id).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_interner_dedups() {
        let mut k = KeyInterner::default();
        let a = k.intern("subsystem");
        let b = k.intern("phase");
        assert_ne!(a, b);
        assert_eq!(k.intern("subsystem"), a); // same name -> same id
        assert_eq!(k.names[a as usize], "subsystem");
        assert_eq!(k.names[b as usize], "phase");
    }

    #[test]
    fn meta_interner_dedups_by_content() {
        let mut m = MetaInterner::default();
        let phys = vec![(0u32, MetaValue::Str("physics".into()))];
        let io = vec![(0u32, MetaValue::Str("io".into()))];
        let id1 = m.intern(&phys);
        let id2 = m.intern(&io);
        assert_ne!(id1, id2); // different value -> different context
        assert_eq!(m.intern(&phys), id1); // identical content -> same id
        assert_eq!(m.contexts[&id1], phys);
        // Dynamic values still dedup per distinct set.
        let req42 = vec![(1u32, MetaValue::Uint(42))];
        let req43 = vec![(1u32, MetaValue::Uint(43))];
        assert_ne!(m.intern(&req42), m.intern(&req43));
        assert_eq!(m.intern(&req42), m.intern(&req42));
    }
}
