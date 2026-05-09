//! Live registry of in-flight runtime resources — channels, sockets, sync
//! primitives. Each cf-tokio shim primitive registers itself when
//! constructed and deregisters on Drop. Each registration carries a
//! `ResourceProbe` impl that the UI can call to read live counters.
//!
//! This is parallel to the task registry: same shape (insert / remove /
//! snapshot), different payload. We deliberately don't model relationships
//! between resources and tasks beyond "task that created me" — backpressure
//! analysis (which tasks are blocked on which channel) is recoverable from
//! per-task `WaitReason`.

use crate::task::TaskId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ResourceId(pub u64);

static NEXT: AtomicU64 = AtomicU64::new(1);

impl ResourceId {
    pub fn fresh() -> Self {
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ResourceKind {
    MpscChannel,
    BroadcastChannel,
    OneshotChannel,
    Notify,
    Semaphore,
    TcpListener,
    TcpStream,
}

impl ResourceKind {
    pub fn label(self) -> &'static str {
        match self {
            ResourceKind::MpscChannel => "mpsc",
            ResourceKind::BroadcastChannel => "broadcast",
            ResourceKind::OneshotChannel => "oneshot",
            ResourceKind::Notify => "Notify",
            ResourceKind::Semaphore => "Semaphore",
            ResourceKind::TcpListener => "TcpListener",
            ResourceKind::TcpStream => "TcpStream",
        }
    }
}

/// Snapshot of a single resource's live state. Fields are optional because
/// not every kind has, say, a queue depth (`Notify` doesn't), and we'd
/// rather have one shape than a sprawling enum.
#[derive(Clone, Debug, Default)]
pub struct ResourceStateSnapshot {
    pub depth: Option<usize>,
    pub capacity: Option<usize>,
    pub peer: Option<String>,
    pub local: Option<String>,
    pub sends: u64,
    pub recvs: u64,
    pub high_water: u64,
    pub closed: bool,
}

pub trait ResourceProbe: Send + Sync {
    fn snapshot(&self) -> ResourceStateSnapshot;
}

#[derive(Clone)]
pub struct ResourceMeta {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub label: String,
    pub created_at: Instant,
    pub created_by: Option<TaskId>,
    pub probe: Arc<dyn ResourceProbe>,
}

#[derive(Clone, Debug)]
pub struct ResourceMetaSnapshot {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub label: String,
    pub age_nanos: u128,
    pub created_by: Option<TaskId>,
    pub state: ResourceStateSnapshot,
}

pub struct ResourceRegistry {
    inner: RwLock<HashMap<ResourceId, ResourceMeta>>,
}

impl ResourceRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(HashMap::new()),
        })
    }

    pub fn insert(
        &self,
        kind: ResourceKind,
        label: String,
        created_by: Option<TaskId>,
        probe: Arc<dyn ResourceProbe>,
    ) -> ResourceId {
        let id = ResourceId::fresh();
        self.inner.write().insert(
            id,
            ResourceMeta {
                id,
                kind,
                label,
                created_at: Instant::now(),
                created_by,
                probe,
            },
        );
        id
    }

    pub fn remove(&self, id: ResourceId) {
        self.inner.write().remove(&id);
    }

    pub fn snapshot(&self) -> Vec<ResourceMetaSnapshot> {
        let g = self.inner.read();
        let mut v: Vec<_> = g
            .values()
            .map(|m| ResourceMetaSnapshot {
                id: m.id,
                kind: m.kind,
                label: m.label.clone(),
                age_nanos: m.created_at.elapsed().as_nanos(),
                created_by: m.created_by,
                state: m.probe.snapshot(),
            })
            .collect();
        v.sort_by_key(|s| s.id.0);
        v
    }
}

/// Shared probe stats useful for channels: thread-safe atomic counters that
/// the resource updates on send/recv, plus a high-water tracker. We keep
/// this in cf-runtime so all shim crates can share the same vocabulary.
#[derive(Default)]
pub struct ChannelCounters {
    pub sends: AtomicU64,
    pub recvs: AtomicU64,
    pub high_water: AtomicU64,
    pub closed: std::sync::atomic::AtomicBool,
}

impl ChannelCounters {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn record_send(&self, depth_after: usize) {
        self.sends.fetch_add(1, Ordering::Relaxed);
        let prev = self.high_water.load(Ordering::Relaxed);
        let now = depth_after as u64;
        if now > prev {
            // Best-effort high-water bump; lossy on contention but cheap.
            let _ = self.high_water.compare_exchange(
                prev,
                now,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }
    pub fn record_recv(&self) {
        self.recvs.fetch_add(1, Ordering::Relaxed);
    }
    pub fn mark_closed(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }
    pub fn read(&self) -> (u64, u64, u64, bool) {
        (
            self.sends.load(Ordering::Relaxed),
            self.recvs.load(Ordering::Relaxed),
            self.high_water.load(Ordering::Relaxed),
            self.closed.load(Ordering::Relaxed),
        )
    }
}
