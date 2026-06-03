//! Lock-free, append-only, **growable** table with stable element
//! addresses.
//!
//! Designed for per-function JIT metadata: written once during
//! compilation, read repeatedly during execution and GC stack
//! walking, never mutated again. Concurrent writers (multiple
//! `extend` calls) and concurrent readers (running JIT code +
//! GC stack scans) coexist safely.
//!
//! ## Layout
//!
//! Conceptually a `Vec<T>` whose storage never reallocates:
//!
//! ```text
//!   bucket[0]: 1   slot
//!   bucket[1]: 2   slots
//!   bucket[2]: 4   slots
//!   bucket[3]: 8   slots
//!     …
//!   bucket[k]: 2^k slots
//! ```
//!
//! For index `n`, `bucket = floor(log2(n + 1))` and slot index within
//! the bucket is `n + 1 - 2^bucket`. Buckets are allocated lazily
//! (the first time a slot in that bucket is written) and never
//! deallocated until the table itself is dropped.
//!
//! With 64 outer slots we cover indices `0..2^64 - 1` — effectively
//! unbounded for any plausible workload, with no hard cap.
//!
//! Memory overhead vs. a perfectly-sized `Vec`: at most ~2× (the
//! last bucket may be half-full).
//!
//! ## Concurrency
//!
//! - `push(value) -> usize` is lock-free. Reserves an index via
//!   `fetch_add`, lazily allocates the target bucket via CAS, then
//!   publishes the value with `OnceLock::set`.
//! - `get(idx) -> Option<&T>` returns a stable reference. The
//!   reference remains valid for the lifetime of `self` because
//!   buckets are never reallocated and the inner `Box`-allocated
//!   slot storage is owned by the bucket.
//! - Multiple `push` and multiple `get` may proceed in parallel.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

const NUM_BUCKETS: usize = 64;

pub struct GrowableTable<T> {
    buckets: [AtomicPtr<Bucket<T>>; NUM_BUCKETS],
    len: AtomicUsize,
}

struct Bucket<T> {
    slots: Box<[OnceLock<T>]>,
}

impl<T> GrowableTable<T> {
    pub fn new() -> Self {
        let buckets: [AtomicPtr<Bucket<T>>; NUM_BUCKETS] =
            std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut()));
        GrowableTable {
            buckets,
            len: AtomicUsize::new(0),
        }
    }

    /// Number of entries that have been published. Read with
    /// Acquire ordering — pairs with the Release publish in `push`.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append `value`, returning its index. The returned index is
    /// stable for the lifetime of the table; subsequent `get(idx)`
    /// calls always see this entry.
    pub fn push(&self, value: T) -> usize {
        let idx = self.len.fetch_add(1, Ordering::AcqRel);
        let (b_idx, slot_idx) = bucket_and_slot(idx);
        let bucket = self.bucket_or_alloc(b_idx);
        bucket.slots[slot_idx]
            .set(value)
            .ok()
            .expect("GrowableTable: slot already initialized (push race)");
        idx
    }

    /// Get a stable reference to the entry at `idx`, or `None` if
    /// `idx >= len()` or the slot's writer hasn't completed yet.
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len.load(Ordering::Acquire) {
            return None;
        }
        let (b_idx, slot_idx) = bucket_and_slot(idx);
        let bucket_ptr = self.buckets[b_idx].load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return None;
        }
        let bucket = unsafe { &*bucket_ptr };
        bucket.slots[slot_idx].get()
    }

    /// Iterate over published entries `[0..len)`. The snapshot of
    /// `len` is taken at iterator construction; entries published
    /// after that point are NOT visible. Yields `&T`.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            table: self,
            cursor: 0,
            end: self.len(),
        }
    }

    fn bucket_or_alloc(&self, b_idx: usize) -> &Bucket<T> {
        let cur = self.buckets[b_idx].load(Ordering::Acquire);
        if !cur.is_null() {
            return unsafe { &*cur };
        }
        // Lazy-allocate this bucket. Bucket k holds 2^k slots
        // (k = 0 → 1 slot, k = 63 → 2^63 slots — though no real
        // workload reaches that).
        let bucket_size: usize = 1 << b_idx;
        let slots: Box<[OnceLock<T>]> = (0..bucket_size).map(|_| OnceLock::new()).collect();
        let new_bucket = Box::into_raw(Box::new(Bucket { slots }));
        match self.buckets[b_idx].compare_exchange(
            std::ptr::null_mut(),
            new_bucket,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => unsafe { &*new_bucket },
            Err(actual) => {
                // Lost the allocation race. Drop ours; use theirs.
                unsafe { drop(Box::from_raw(new_bucket)) };
                unsafe { &*actual }
            }
        }
    }
}

impl<T> Default for GrowableTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for GrowableTable<T> {
    fn drop(&mut self) {
        for slot in &self.buckets {
            let p = slot.load(Ordering::Acquire);
            if !p.is_null() {
                unsafe { drop(Box::from_raw(p)) };
            }
        }
    }
}

// SAFETY: GrowableTable<T> is Send/Sync if T is. All internal
// mutation is guarded by atomics and OnceLock; cross-thread reads
// see fully-initialized entries thanks to the AcqRel/Acquire
// ordering on `len` and `buckets`.
unsafe impl<T: Send> Send for GrowableTable<T> {}
unsafe impl<T: Send + Sync> Sync for GrowableTable<T> {}

pub struct Iter<'a, T> {
    table: &'a GrowableTable<T>,
    cursor: usize,
    end: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.cursor >= self.end {
            return None;
        }
        let item = self.table.get(self.cursor);
        self.cursor += 1;
        item
    }
}

/// Decompose `idx` into `(bucket, slot)`. Bucket `k` holds `2^k`
/// slots; bucket 0 holds index 0; bucket 1 holds 1..=2; bucket 2
/// holds 3..=6; etc.
fn bucket_and_slot(idx: usize) -> (usize, usize) {
    let n = idx + 1;
    let bucket = (usize::BITS - 1 - n.leading_zeros()) as usize;
    let bucket_start = (1usize << bucket) - 1;
    (bucket, idx - bucket_start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn decomposition() {
        assert_eq!(bucket_and_slot(0), (0, 0));
        assert_eq!(bucket_and_slot(1), (1, 0));
        assert_eq!(bucket_and_slot(2), (1, 1));
        assert_eq!(bucket_and_slot(3), (2, 0));
        assert_eq!(bucket_and_slot(6), (2, 3));
        assert_eq!(bucket_and_slot(7), (3, 0));
        assert_eq!(bucket_and_slot(14), (3, 7));
        assert_eq!(bucket_and_slot(15), (4, 0));
    }

    #[test]
    fn push_and_get() {
        let t: GrowableTable<u32> = GrowableTable::new();
        for i in 0..1000 {
            let idx = t.push(i * 10);
            assert_eq!(idx, i as usize);
        }
        for i in 0..1000 {
            assert_eq!(t.get(i as usize), Some(&(i * 10)));
        }
        assert_eq!(t.get(1000), None);
        assert_eq!(t.len(), 1000);
    }

    #[test]
    fn iter() {
        let t: GrowableTable<u32> = GrowableTable::new();
        for i in 0..50 {
            t.push(i);
        }
        let collected: Vec<u32> = t.iter().copied().collect();
        assert_eq!(collected, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn references_are_stable() {
        // Capture a reference to early entries, then push lots more,
        // then verify the early references are still valid.
        let t: GrowableTable<u32> = GrowableTable::new();
        t.push(42);
        let r0 = t.get(0).unwrap() as *const u32;
        for i in 0..10_000 {
            t.push(i);
        }
        let r0_again = t.get(0).unwrap() as *const u32;
        assert_eq!(r0, r0_again, "reference at idx 0 must be stable");
        assert_eq!(*unsafe { &*r0 }, 42);
    }

    #[test]
    fn concurrent_pushers() {
        let t: Arc<GrowableTable<u32>> = Arc::new(GrowableTable::new());
        let mut handles = Vec::new();
        for thread_id in 0..4 {
            let t = t.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    t.push(thread_id * 10000 + i);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(t.len(), 4000);
        let mut all: Vec<u32> = t.iter().copied().collect();
        all.sort();
        // Each thread's range should be present.
        for thread_id in 0..4u32 {
            for i in 0..1000u32 {
                let v = thread_id * 10000 + i;
                assert!(all.binary_search(&v).is_ok(), "missing {v}");
            }
        }
    }

    #[test]
    fn concurrent_pushers_and_readers() {
        // Pushers and readers run simultaneously. Readers must never
        // see a torn entry — every successful `get` returns a
        // fully-initialized value.
        let t: Arc<GrowableTable<u64>> = Arc::new(GrowableTable::new());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut handles = Vec::new();

        // Pushers.
        for tid in 0..3u64 {
            let t = t.clone();
            handles.push(thread::spawn(move || {
                for i in 0..2000u64 {
                    let val = (tid << 32) | i;
                    t.push(val);
                }
            }));
        }

        // Readers.
        for _ in 0..2 {
            let t = t.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let len = t.len();
                    for i in 0..len {
                        if let Some(&v) = t.get(i) {
                            // Verify the high bits are a valid tid (0..3).
                            let tid = v >> 32;
                            assert!(tid < 3, "torn read: tid={tid} val={v}");
                        }
                    }
                }
            }));
        }

        // Let pushers run to completion.
        std::thread::sleep(std::time::Duration::from_millis(20));
        stop.store(true, Ordering::Release);

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(t.len(), 6000);
    }
}
