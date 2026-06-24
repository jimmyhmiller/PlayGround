//! Lock-free event ring — the hot-path sink for allocation/free records.
//!
//! Design: a fixed `[Slot]` (capacity a power of two) plus a monotonic `head`
//! claim counter (the Disruptor / per-slot-seqlock pattern). A producer claims a
//! global sequence number, writes its [`RawEvent`] into `slot[seq % cap]`, then
//! publishes by storing `seq` into that slot's `seq` stamp with Release. The
//! claim order *is* the total order of events — which is what lets a consumer
//! pair `alloc(A)` with a later `free(A)` even across threads and address reuse.
//!
//! Two modes (see [`RingMode`]):
//! * **Overwrite** — producers are wait-free and never block; when the ring is
//!   full they lap the consumer, overwriting the oldest unread slots. The
//!   consumer detects the lap (its read cursor fell more than `cap` behind
//!   `head`) and counts the gap. A per-read seqlock check guards against reading
//!   a slot mid-overwrite.
//! * **Reliable** — a producer that would overwrite an unread slot first spins
//!   (bounded) for the consumer to catch up; only if the consumer is truly stuck
//!   does it drop-and-count, so a runaway producer can never deadlock the app.
//!
//! Single consumer: exactly one thread may call [`Ring::drain`] (the pump). Many
//! producers may [`Ring::push`] concurrently.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use memscope_proto::{EventKind, RawEvent, SiteId};

/// Producer policy when the ring is full.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum RingMode {
    /// Never block; overwrite the oldest unread events (consumer detects loss).
    Overwrite = 0,
    /// Apply bounded backpressure so the consumer doesn't lose events.
    Reliable = 1,
}

impl RingMode {
    fn from_u8(v: u8) -> RingMode {
        match v {
            1 => RingMode::Reliable,
            _ => RingMode::Overwrite,
        }
    }
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
    /// The sequence number of the event currently in `cell`, or [`UNPUBLISHED`]
    /// before anything has been published to this slot.
    seq: AtomicU64,
    cell: UnsafeCell<RawEvent>,
}

pub struct Ring {
    slots: Box<[Slot]>,
    mask: u64,
    capacity: u64,
    /// Next sequence number to be claimed by a producer (== total claimed).
    head: AtomicU64,
    /// Next sequence number the (single) consumer will read; published so
    /// producers can apply backpressure.
    consumer_pos: AtomicU64,
    dropped: AtomicU64,
    mode: AtomicU8,
    /// Backpressure spin budget before a Reliable producer gives up and drops.
    spin_limit: u32,
}

// SAFETY: cross-thread access to `cell` is synchronized through the `seq` stamp
// (Release on publish / Acquire on read) and the global `head`/`consumer_pos`
// counters — the seqlock protocol. `RawEvent` is `Copy` POD.
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    /// Create a ring holding at least `capacity` events (rounded up to a power
    /// of two).
    pub fn new(capacity: usize, mode: RingMode) -> Ring {
        let cap = capacity.next_power_of_two().max(2) as u64;
        let slots = (0..cap)
            .map(|_| Slot {
                seq: AtomicU64::new(UNPUBLISHED),
                cell: UnsafeCell::new(EMPTY_EVENT),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ring {
            slots,
            mask: cap - 1,
            capacity: cap,
            head: AtomicU64::new(0),
            consumer_pos: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            mode: AtomicU8::new(mode as u8),
            spin_limit: 256,
        }
    }

    #[allow(dead_code)] // used by tests + future consumers
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn mode(&self) -> RingMode {
        RingMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    pub fn set_mode(&self, mode: RingMode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Push one event (any number of concurrent producers). Wait-free in
    /// Overwrite mode; in Reliable mode may spin briefly under backpressure.
    #[inline]
    pub fn push(&self, ev: RawEvent) {
        let seq = match self.mode() {
            RingMode::Overwrite => self.head.fetch_add(1, Ordering::Relaxed),
            RingMode::Reliable => match self.claim_reliable() {
                Some(seq) => seq,
                None => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            },
        };
        let slot = &self.slots[(seq & self.mask) as usize];
        // SAFETY: we exclusively own this slot for sequence `seq` until we
        // publish below; a concurrent consumer guards reads with the seqlock.
        unsafe {
            *slot.cell.get() = ev;
        }
        slot.seq.store(seq, Ordering::Release);
    }

    /// Reliable claim: CAS the head forward only once there's room, spinning a
    /// bounded number of times. Returns `None` if the consumer stayed stuck.
    #[inline]
    fn claim_reliable(&self) -> Option<u64> {
        let mut spins = 0u32;
        loop {
            let head = self.head.load(Ordering::Relaxed);
            let cons = self.consumer_pos.load(Ordering::Acquire);
            if head - cons >= self.capacity {
                // Full — wait for the consumer.
                if spins >= self.spin_limit {
                    return None;
                }
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            if self
                .head
                .compare_exchange_weak(head, head + 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Some(head);
            }
        }
    }

    /// Drain newly-available events into `out` in sequence order. Must be called
    /// by a single consumer thread. Returns the number drained.
    pub fn drain(&self, out: &mut Vec<RawEvent>) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let mut pos = self.consumer_pos.load(Ordering::Relaxed);

        // Lap detection: anything older than `head - capacity` is gone.
        let oldest = head.saturating_sub(self.capacity);
        if pos < oldest {
            self.dropped.fetch_add(oldest - pos, Ordering::Relaxed);
            pos = oldest;
        }

        let start = pos;
        while pos < head {
            let slot = &self.slots[(pos & self.mask) as usize];
            let s1 = slot.seq.load(Ordering::Acquire);
            if s1 != pos {
                // Either not yet published (producer claimed but hasn't stored)
                // — stop at the contiguous prefix — or already overwritten by a
                // lapping producer (s1 > pos); count it lost and move on.
                if s1 != UNPUBLISHED && s1 > pos {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    pos += 1;
                    continue;
                }
                break;
            }
            // SAFETY: `RawEvent` is `Copy`; the seqlock recheck below discards a
            // read that raced an overwrite.
            let ev = unsafe { *slot.cell.get() };
            let s2 = slot.seq.load(Ordering::Acquire);
            if s2 != pos {
                // Overwritten while we copied (Overwrite mode only). Drop it.
                self.dropped.fetch_add(1, Ordering::Relaxed);
                pos += 1;
                continue;
            }
            out.push(ev);
            pos += 1;
        }

        self.consumer_pos.store(pos, Ordering::Release);
        (pos - start) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(seq: u64) -> RawEvent {
        RawEvent {
            kind: EventKind::Alloc,
            seq,
            ts_nanos: 0,
            addr: seq, // use addr to carry an identity we can assert on
            size: 0,
            align: 0,
            site: SiteId::NONE,
            thread: 0,
        }
    }

    #[test]
    fn reliable_roundtrip_in_order_no_loss() {
        let ring = Ring::new(1024, RingMode::Reliable);
        for i in 0..500 {
            ring.push(ev(i));
        }
        let mut out = Vec::new();
        let n = ring.drain(&mut out);
        assert_eq!(n, 500);
        assert_eq!(ring.dropped(), 0);
        for (i, e) in out.iter().enumerate() {
            assert_eq!(e.addr, i as u64);
        }
    }

    #[test]
    fn drain_then_more() {
        let ring = Ring::new(64, RingMode::Reliable);
        for i in 0..40 {
            ring.push(ev(i));
        }
        let mut out = Vec::new();
        assert_eq!(ring.drain(&mut out), 40);
        out.clear();
        for i in 40..60 {
            ring.push(ev(i));
        }
        assert_eq!(ring.drain(&mut out), 20);
        assert_eq!(out[0].addr, 40);
        assert_eq!(ring.dropped(), 0);
    }

    #[test]
    fn overwrite_keeps_newest_and_counts_drops() {
        let cap = 64usize;
        let ring = Ring::new(cap, RingMode::Overwrite);
        let total = (cap + 10) as u64;
        for i in 0..total {
            ring.push(ev(i));
        }
        let mut out = Vec::new();
        let n = ring.drain(&mut out);
        // We keep the most recent `cap` events; the first 10 were overwritten.
        assert_eq!(n, cap);
        assert_eq!(ring.dropped(), 10);
        assert_eq!(out.first().unwrap().addr, 10);
        assert_eq!(out.last().unwrap().addr, total - 1);
    }

    #[test]
    fn reliable_backpressure_then_drop_when_consumer_never_runs() {
        // Capacity 8, never drain -> after 8 the producer spins out and drops.
        let ring = Ring::new(8, RingMode::Reliable);
        for i in 0..100 {
            ring.push(ev(i));
        }
        // Exactly `capacity` made it in; the rest were dropped (consumer stuck).
        let mut out = Vec::new();
        let n = ring.drain(&mut out);
        assert_eq!(n as u64, ring.capacity());
        assert_eq!(ring.dropped(), 100 - ring.capacity());
    }

    #[test]
    fn concurrent_producers_reliable_lossless() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let ring = Arc::new(Ring::new(1 << 16, RingMode::Reliable));
        let stop = Arc::new(AtomicBool::new(false));
        let per_thread = 50_000u64;
        let threads = 4u64;

        // A consumer thread draining concurrently so the ring never fills.
        let collected = Arc::new(std::sync::Mutex::new(Vec::new()));
        let consumer = {
            let ring = ring.clone();
            let stop = stop.clone();
            let collected = collected.clone();
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    let n = ring.drain(&mut buf);
                    if n > 0 {
                        collected.lock().unwrap().extend(buf.iter().map(|e| e.addr));
                    } else if stop.load(Ordering::Acquire) {
                        // final drain
                        buf.clear();
                        ring.drain(&mut buf);
                        collected.lock().unwrap().extend(buf.iter().map(|e| e.addr));
                        break;
                    }
                }
            })
        };

        let producers: Vec<_> = (0..threads)
            .map(|t| {
                let ring = ring.clone();
                std::thread::spawn(move || {
                    for i in 0..per_thread {
                        ring.push(ev(t * per_thread + i));
                    }
                })
            })
            .collect();
        for p in producers {
            p.join().unwrap();
        }
        // Give the consumer a moment, then stop.
        std::thread::sleep(std::time::Duration::from_millis(50));
        stop.store(true, Ordering::Release);
        consumer.join().unwrap();

        assert_eq!(ring.dropped(), 0, "reliable mode must not drop");
        let mut got = Arc::try_unwrap(collected).unwrap().into_inner().unwrap();
        got.sort_unstable();
        assert_eq!(got.len() as u64, threads * per_thread);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, i as u64, "every event delivered exactly once");
        }
    }
}
