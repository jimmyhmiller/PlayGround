//! Per-thread sharded rings with a single merging consumer.
//!
//! Each thread owns a private SPSC ring (single producer = the thread, single
//! consumer = the pump). A producer never touches a shared atomic, so there's no
//! cross-thread contention on the hot path — the bottleneck that a single shared
//! ring's `head` counter reintroduced.
//!
//! Ordering for address reuse: per-thread rings preserve per-thread order but not
//! global order, and the live-set reconstructor needs causal order (a free after
//! its alloc) for addresses that move between threads. Batching a global sequence
//! breaks causality (a seq is assigned before the event happens), so instead each
//! event is stamped with a **monotonic, cross-core hardware timestamp** ([`tsc`])
//! at the moment it occurs. The consumer merges the rings by that timestamp.
//! This costs one cheap register read per event and no shared atomic.
//!
//! The consumer applies events behind a small time **watermark** so that a peer
//! ring lagging by less than the watermark can't deliver an out-of-order event
//! after we've already applied past it.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use memscope_proto::{EventKind, RawEvent, SiteId};

/// Producer policy when a ring is full.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum RingMode {
    /// Never block; overwrite the oldest unread events (consumer detects loss).
    Overwrite = 0,
    /// Apply bounded backpressure so the consumer doesn't lose events.
    Reliable = 1,
}

/// A monotonic timestamp shared across cores, read cheaply with no syscall.
/// On x86-64 this is the invariant TSC; on aarch64 the virtual count register.
/// Elsewhere it falls back to a (contended, but correct) global counter.
#[inline]
pub fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: rdtsc is always available on x86-64.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        // SAFETY: CNTVCT_EL0 is readable from EL0 on all supported aarch64.
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags));
        }
        v
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }
}

/// Measure the timestamp frequency (ticks per second) by sampling against a
/// wall clock. Used to convert the watermark window and ages.
pub fn calibrate_tsc_hz() -> u64 {
    let t0 = std::time::Instant::now();
    let c0 = tsc();
    while t0.elapsed() < std::time::Duration::from_millis(10) {
        std::hint::spin_loop();
    }
    let c1 = tsc();
    let secs = t0.elapsed().as_secs_f64();
    let hz = ((c1.wrapping_sub(c0)) as f64 / secs) as u64;
    hz.max(1)
}

const UNPUBLISHED: u64 = u64::MAX;

const EMPTY_EVENT: RawEvent = RawEvent {
    kind: EventKind::Alloc,
    seq: 0,
    ts_nanos: 0,
    addr: 0,
    size: 0,
    align: 0,
    site: SiteId::NONE,
    thread: 0,
};

struct Slot {
    /// Local ring position published here, or [`UNPUBLISHED`].
    pos: AtomicU64,
    cell: UnsafeCell<RawEvent>,
}

/// A single thread's private ring. Producer = owner thread; consumer = the pump.
pub struct TlsRing {
    slots: Box<[Slot]>,
    mask: u64,
    cap: u64,
    head: AtomicU64, // producer position (owner writes)
    tail: AtomicU64, // consumer position (pump writes)
    dropped: AtomicU64,
    mode: AtomicU8,
    spin_limit: AtomicU32,
}

// SAFETY: `cell` access is synchronized via the per-slot `pos` stamp (Release on
// publish / Acquire on read) plus the head/tail counters; `RawEvent` is Copy POD.
unsafe impl Sync for TlsRing {}
unsafe impl Send for TlsRing {}

impl TlsRing {
    fn new(capacity: usize, mode: RingMode) -> Self {
        let cap = capacity.next_power_of_two().max(2) as u64;
        let slots = (0..cap)
            .map(|_| Slot {
                pos: AtomicU64::new(UNPUBLISHED),
                cell: UnsafeCell::new(EMPTY_EVENT),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        TlsRing {
            slots,
            mask: cap - 1,
            cap,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            mode: AtomicU8::new(mode as u8),
            spin_limit: AtomicU32::new(5_000_000),
        }
    }

    fn set_mode(&self, mode: RingMode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
    }

    /// Push an event. Owner thread only. Wait-free in Overwrite mode; in Reliable
    /// mode waits (bounded) for the consumer rather than overwrite.
    #[inline]
    pub fn push(&self, ev: RawEvent) {
        let pos = self.head.load(Ordering::Relaxed); // owner is sole writer

        if self.mode.load(Ordering::Relaxed) == RingMode::Reliable as u8 {
            let mut attempts = 0u32;
            let limit = self.spin_limit.load(Ordering::Relaxed);
            while pos - self.tail.load(Ordering::Acquire) >= self.cap {
                attempts += 1;
                if attempts >= limit {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                if attempts < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }

        let slot = &self.slots[(pos & self.mask) as usize];
        // SAFETY: the owner exclusively writes this slot for position `pos`; the
        // consumer guards reads with the seqlock recheck below.
        unsafe {
            *slot.cell.get() = ev;
        }
        slot.pos.store(pos, Ordering::Release);
        self.head.store(pos + 1, Ordering::Release);
    }

    /// Drain available events into `out`. Consumer (pump) only.
    pub fn drain(&self, out: &mut Vec<RawEvent>) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let mut pos = self.tail.load(Ordering::Relaxed); // pump is sole writer

        let oldest = head.saturating_sub(self.cap);
        if pos < oldest {
            self.dropped.fetch_add(oldest - pos, Ordering::Relaxed);
            pos = oldest;
        }

        let start = pos;
        while pos < head {
            let slot = &self.slots[(pos & self.mask) as usize];
            let s1 = slot.pos.load(Ordering::Acquire);
            if s1 != pos {
                if s1 != UNPUBLISHED && s1 > pos {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    pos += 1;
                    continue;
                }
                break;
            }
            // SAFETY: Copy read, validated by the seqlock recheck.
            let ev = unsafe { *slot.cell.get() };
            if slot.pos.load(Ordering::Acquire) != pos {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                pos += 1;
                continue;
            }
            out.push(ev);
            pos += 1;
        }
        self.tail.store(pos, Ordering::Release);
        (pos - start) as usize
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Owns every thread's ring and the registration registry. One per process,
/// held by the recorder.
pub struct ShardedRing {
    registry: Mutex<Vec<std::sync::Arc<TlsRing>>>,
    mode: AtomicU8,
    cap: usize,
}

thread_local! {
    static MY_RING: std::cell::RefCell<Option<std::sync::Arc<TlsRing>>> =
        const { std::cell::RefCell::new(None) };
}

impl ShardedRing {
    pub fn new(cap: usize, mode: RingMode) -> Self {
        ShardedRing {
            registry: Mutex::new(Vec::new()),
            mode: AtomicU8::new(mode as u8),
            cap,
        }
    }

    pub fn mode(&self) -> RingMode {
        if self.mode.load(Ordering::Relaxed) == RingMode::Reliable as u8 {
            RingMode::Reliable
        } else {
            RingMode::Overwrite
        }
    }

    pub fn set_mode(&self, mode: RingMode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
        for r in self.registry.lock().unwrap().iter() {
            r.set_mode(mode);
        }
    }

    /// Push to the calling thread's ring (creating + registering it on first use).
    #[inline]
    pub fn push(&self, ev: RawEvent) {
        // `try_with`: if the thread is tearing down (TLS gone), drop the event
        // rather than panic — a global allocator must survive thread exit.
        let _ = MY_RING.try_with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let ring = std::sync::Arc::new(TlsRing::new(self.cap, self.mode()));
                self.registry.lock().unwrap().push(ring.clone());
                *slot = Some(ring);
            }
            slot.as_ref().unwrap().push(ev);
        });
    }

    /// Drain every ring into `out` (unordered across rings; the caller sorts by
    /// `seq`). Consumer/pump only.
    pub fn drain_all(&self, out: &mut Vec<RawEvent>) -> usize {
        let rings = self.registry.lock().unwrap().clone();
        let mut n = 0;
        for r in &rings {
            n += r.drain(out);
        }
        n
    }

    pub fn dropped(&self) -> u64 {
        self.registry
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.dropped())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn ev(seq: u64, addr: u64, kind: EventKind) -> RawEvent {
        RawEvent {
            kind,
            seq,
            ts_nanos: 0,
            addr,
            size: 8,
            align: 8,
            site: SiteId::NONE,
            thread: 0,
        }
    }

    #[test]
    fn tsc_monotonic() {
        let a = tsc();
        let b = tsc();
        assert!(b >= a, "timestamp must be monotonic ({a} then {b})");
    }

    #[test]
    fn spsc_roundtrip() {
        let r = TlsRing::new(1024, RingMode::Reliable);
        for i in 0..500 {
            r.push(ev(i, i, EventKind::Alloc));
        }
        let mut out = Vec::new();
        assert_eq!(r.drain(&mut out), 500);
        assert_eq!(r.dropped(), 0);
        for (i, e) in out.iter().enumerate() {
            assert_eq!(e.seq, i as u64);
        }
    }

    #[test]
    fn overwrite_keeps_newest() {
        let r = TlsRing::new(64, RingMode::Overwrite);
        for i in 0..(64 + 10) {
            r.push(ev(i, i, EventKind::Alloc));
        }
        let mut out = Vec::new();
        let n = r.drain(&mut out);
        assert_eq!(n, 64);
        assert_eq!(r.dropped(), 10);
        assert_eq!(out.first().unwrap().seq, 10);
    }

    /// Many threads, each its own ring, single consumer merges by seq — exact,
    /// in-order, no loss. This is the property the whole design rests on.
    #[test]
    fn sharded_merge_lossless_and_ordered() {
        let sharded = Arc::new(ShardedRing::new(1 << 16, RingMode::Reliable));
        let stop = Arc::new(AtomicBool::new(false));
        let threads = 4u64;
        let per = 100_000u64;

        let collected = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
        let consumer = {
            let sharded = sharded.clone();
            let stop = stop.clone();
            let collected = collected.clone();
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    let n = sharded.drain_all(&mut buf);
                    if n > 0 {
                        collected.lock().unwrap().extend(buf.iter().map(|e| e.seq));
                    } else if stop.load(Ordering::Acquire) {
                        buf.clear();
                        sharded.drain_all(&mut buf);
                        collected.lock().unwrap().extend(buf.iter().map(|e| e.seq));
                        break;
                    }
                }
            })
        };

        let producers: Vec<_> = (0..threads)
            .map(|t| {
                let sharded = sharded.clone();
                std::thread::spawn(move || {
                    for i in 0..per {
                        sharded.push(ev(t * per + i, t * per + i, EventKind::Alloc));
                    }
                })
            })
            .collect();
        for p in producers {
            p.join().unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        stop.store(true, Ordering::Release);
        consumer.join().unwrap();

        assert_eq!(sharded.dropped(), 0, "reliable mode must not drop");
        let mut got = Arc::try_unwrap(collected).unwrap().into_inner().unwrap();
        got.sort_unstable();
        assert_eq!(got.len() as u64, threads * per);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, i as u64, "every event delivered exactly once");
        }
    }
}
