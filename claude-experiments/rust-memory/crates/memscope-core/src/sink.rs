//! Pluggable consumers for the event stream.
//!
//! The hot path only appends [`RawEvent`]s to the lock-free [`crate::ring::Ring`].
//! A single *pump* thread drains that ring and feeds the batches to an
//! [`EventSink`] — which is where pluggability lives. A sink can:
//! * reconstruct the live set in memory (so `snapshot()` works) — [`LiveSet`];
//! * write raw events to a file / socket / network for posthoc replay;
//! * fan out to several of the above ([`FanOut`]);
//! * be an arbitrary closure ([`FnSink`]).
//!
//! Exactly one pump (hence one consumer) drains the ring; install a [`FanOut`]
//! to feed multiple destinations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use memscope_proto::{EventKind, RawEvent};

use crate::recorder::recorder;

/// A consumer of drained allocation events. Implementations receive batches in
/// strict sequence order.
pub trait EventSink: Send {
    /// Handle a batch of events (sequence-ordered).
    fn consume(&mut self, events: &[RawEvent]);
    /// Called when the ring's cumulative drop count changes (Overwrite mode fell
    /// behind, or Reliable backpressure gave up). The reconstruction is lossy
    /// from this point.
    fn on_drops(&mut self, _total_dropped: u64) {}
    /// Final flush when the pump stops.
    fn flush(&mut self) {}
}

/// A sink backed by a closure (handy for tests / quick taps).
pub struct FnSink<F: FnMut(&[RawEvent]) + Send>(pub F);

impl<F: FnMut(&[RawEvent]) + Send> EventSink for FnSink<F> {
    fn consume(&mut self, events: &[RawEvent]) {
        (self.0)(events)
    }
}

/// Fan one stream out to several sinks.
pub struct FanOut(pub Vec<Box<dyn EventSink>>);

impl EventSink for FanOut {
    fn consume(&mut self, events: &[RawEvent]) {
        for s in self.0.iter_mut() {
            s.consume(events);
        }
    }
    fn on_drops(&mut self, total: u64) {
        for s in self.0.iter_mut() {
            s.on_drops(total);
        }
    }
    fn flush(&mut self) {
        for s in self.0.iter_mut() {
            s.flush();
        }
    }
}

/// Reconstructs the live allocation set by replaying events off the hot path —
/// the consumer-side equivalent of the in-process live table. Replaying in
/// sequence order makes `alloc(A) … free(A) … alloc(A)` (address reuse) correct.
#[derive(Default)]
pub struct LiveSet {
    live: HashMap<u64, LiveRec>,
    live_bytes: u64,
    total_allocs: u64,
    total_alloc_bytes: u64,
    dropped: u64,
}

/// What the reconstructor remembers per live allocation.
#[derive(Clone, Copy, Debug)]
pub struct LiveRec {
    pub size: u64,
    pub site: memscope_proto::SiteId,
    pub thread: u32,
    pub ts_nanos: u64,
}

impl LiveSet {
    pub fn new() -> Self {
        LiveSet::default()
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }
    pub fn total_allocs(&self) -> u64 {
        self.total_allocs
    }
    pub fn total_alloc_bytes(&self) -> u64 {
        self.total_alloc_bytes
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Iterate the current live allocations.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &LiveRec)> {
        self.live.iter().map(|(&a, r)| (a, r))
    }

    fn apply(&mut self, e: &RawEvent) {
        match e.kind {
            EventKind::Alloc | EventKind::ReallocGrow => {
                self.live.insert(
                    e.addr,
                    LiveRec {
                        size: e.size,
                        site: e.site,
                        thread: e.thread,
                        ts_nanos: e.ts_nanos,
                    },
                );
                self.live_bytes += e.size;
                self.total_allocs += 1;
                self.total_alloc_bytes += e.size;
            }
            EventKind::Dealloc => {
                if let Some(rec) = self.live.remove(&e.addr) {
                    self.live_bytes = self.live_bytes.saturating_sub(rec.size);
                }
            }
        }
    }
}

impl EventSink for LiveSet {
    fn consume(&mut self, events: &[RawEvent]) {
        for e in events {
            self.apply(e);
        }
    }
    fn on_drops(&mut self, total: u64) {
        self.dropped = total;
    }
}

/// A shared reconstructor: the pump updates it while another thread reads its
/// current state. (Local trait on a foreign type — allowed by the orphan rule.)
impl EventSink for Arc<std::sync::Mutex<LiveSet>> {
    fn consume(&mut self, events: &[RawEvent]) {
        self.lock().unwrap().consume(events);
    }
    fn on_drops(&mut self, total: u64) {
        self.lock().unwrap().on_drops(total);
    }
    fn flush(&mut self) {
        self.lock().unwrap().flush();
    }
}

/// Handle to a running pump. Drop or call [`Consumer::stop`] to wind it down.
pub struct Consumer {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<Box<dyn EventSink>>>,
}

impl Consumer {
    /// Stop the pump and return the sink (final state, e.g. a populated
    /// [`LiveSet`]). Performs a final drain first.
    pub fn stop(mut self) -> Box<dyn EventSink> {
        self.stop.store(true, Ordering::Release);
        self.join.take().unwrap().join().unwrap()
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            self.stop.store(true, Ordering::Release);
            let _ = join.join();
        }
    }
}

/// Spawn the single pump thread that drains the global ring and feeds `sink`.
/// The pump thread excludes itself from allocation tracking.
pub fn spawn_consumer(mut sink: Box<dyn EventSink>, poll: Duration) -> Consumer {
    // The ring is only fed while streaming is on; a consumer implies it.
    recorder().set_event_streaming(true);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let join = std::thread::Builder::new()
        .name("memscope-consumer".into())
        .spawn(move || {
            crate::exclude_current_thread();
            let mut buf: Vec<RawEvent> = Vec::with_capacity(8192);
            let mut last_dropped = 0u64;
            loop {
                buf.clear();
                let n = recorder().drain_ring(&mut buf);
                if n > 0 {
                    sink.consume(&buf);
                }
                let d = recorder().ring_dropped();
                if d != last_dropped {
                    sink.on_drops(d);
                    last_dropped = d;
                }
                if n == 0 {
                    if stop_thread.load(Ordering::Acquire) {
                        // One last drain to catch anything published during shutdown.
                        buf.clear();
                        if recorder().drain_ring(&mut buf) > 0 {
                            sink.consume(&buf);
                        }
                        sink.flush();
                        break;
                    }
                    std::thread::sleep(poll);
                }
            }
            sink
        })
        .expect("failed to spawn memscope consumer thread");

    Consumer {
        stop,
        join: Some(join),
    }
}
