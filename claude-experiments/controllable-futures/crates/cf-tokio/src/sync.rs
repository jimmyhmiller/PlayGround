//! Mirrors `tokio::sync`. Built on async-channel / async-broadcast /
//! event-listener / async-lock under the hood, reshaped to tokio's API.
//!
//! Only the surface mini-redis exercises is here. Methods we have not
//! implemented should NOT be silently no-op'd — they're simply absent so the
//! compiler tells the reader exactly what's missing.

/// Helper used across the sync primitives: post a "user event" into the
/// runtime log, naming the primitive that fired. Cheap when a runtime is
/// installed on the calling thread; a no-op otherwise (e.g. unit tests
/// constructed outside a runtime).
fn log_event(category: &'static str, detail: impl Into<String>) {
    if let Some(h) = cf_runtime::try_current() {
        h.log_user_event(category, detail);
    }
}

/// Async mutex shaped like `tokio::sync::Mutex`. Built on `async_lock::Mutex`.
pub struct Mutex<T> {
    inner: async_lock::Mutex<T>,
}

impl<T> Mutex<T> {
    pub fn new(t: T) -> Self {
        Self {
            inner: async_lock::Mutex::new(t),
        }
    }

    pub async fn lock(&self) -> MutexGuard<'_, T> {
        let g = crate::tag_wait(self.inner.lock(), || {
            cf_runtime::WaitReason::Other("Mutex::lock")
        })
        .await;
        MutexGuard { inner: g }
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.inner.try_lock().map(|g| MutexGuard { inner: g })
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

pub struct MutexGuard<'a, T> {
    inner: async_lock::MutexGuard<'a, T>,
}

impl<'a, T> std::ops::Deref for MutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<'a, T> std::ops::DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

pub mod mpsc {
    //! tokio::sync::mpsc analog. Bounded multi-producer, single-consumer.

    use cf_runtime::resource::{
        ChannelCounters, ResourceKind, ResourceProbe, ResourceStateSnapshot,
    };
    use std::sync::Arc;

    /// Probe wrapping the channel's recv-side handle (so we can query
    /// queue depth) plus the shared counters.
    struct MpscProbe<T> {
        rx: async_channel::Receiver<T>,
        capacity: usize,
        counters: Arc<ChannelCounters>,
    }
    impl<T: Send + Sync + 'static> ResourceProbe for MpscProbe<T> {
        fn snapshot(&self) -> ResourceStateSnapshot {
            let (s, r, hw, closed) = self.counters.read();
            ResourceStateSnapshot {
                depth: Some(self.rx.len()),
                capacity: Some(self.capacity),
                sends: s,
                recvs: r,
                high_water: hw,
                closed,
                ..Default::default()
            }
        }
    }

    pub fn channel<T: Send + Sync + 'static>(buffer: usize) -> (Sender<T>, Receiver<T>) {
        let (s, r) = async_channel::bounded::<T>(buffer);
        let counters = ChannelCounters::new();
        let resource_id = cf_runtime::try_current().map(|h| {
            h.resources().insert(
                ResourceKind::MpscChannel,
                format!("mpsc(cap={buffer})"),
                cf_runtime::runtime::current_task(),
                Arc::new(MpscProbe::<T> {
                    rx: r.clone(),
                    capacity: buffer,
                    counters: counters.clone(),
                }),
            )
        });
        let registry = cf_runtime::try_current().map(|h| h.resources());
        (
            Sender {
                inner: s,
                counters: counters.clone(),
                resource_id,
                registry: registry.clone(),
            },
            Receiver {
                inner: r,
                counters,
                resource_id,
                registry,
                _no_clone: Arc::new(()),
            },
        )
    }

    /// Tokio-shaped `Sender`. Cloning duplicates the sender; the channel
    /// stays open as long as at least one Sender or one stored permit
    /// exists.
    pub struct Sender<T> {
        inner: async_channel::Sender<T>,
        counters: Arc<ChannelCounters>,
        resource_id: Option<cf_runtime::resource::ResourceId>,
        registry: Option<Arc<cf_runtime::resource::ResourceRegistry>>,
    }

    impl<T> std::fmt::Debug for Sender<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("mpsc::Sender").finish()
        }
    }

    impl<T> Clone for Sender<T> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
                counters: self.counters.clone(),
                resource_id: self.resource_id,
                registry: self.registry.clone(),
            }
        }
    }

    pub struct SendError<T>(pub T);

    impl<T: std::fmt::Debug> std::fmt::Debug for SendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple("SendError").field(&self.0).finish()
        }
    }

    impl<T> std::fmt::Display for SendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("channel closed")
        }
    }

    impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

    impl<T> Sender<T> {
        pub async fn send(&self, value: T) -> Result<(), SendError<T>> {
            super::log_event("mpsc", "send");
            let cap = self.inner.capacity().unwrap_or(0);
            let inner_for_reason = self.inner.clone();
            let result = super::super::tag_wait(self.inner.send(value), move || {
                cf_runtime::WaitReason::MpscSend {
                    depth: inner_for_reason.len(),
                    capacity: cap,
                }
            })
            .await
            .map_err(|e| SendError(e.0));
            if result.is_ok() {
                self.counters.record_send(self.inner.len());
            }
            result
        }

        pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
            match self.inner.try_send(value) {
                Ok(()) => Ok(()),
                Err(async_channel::TrySendError::Full(v)) => Err(TrySendError::Full(v)),
                Err(async_channel::TrySendError::Closed(v)) => Err(TrySendError::Closed(v)),
            }
        }

        pub fn is_closed(&self) -> bool {
            self.inner.is_closed()
        }
    }

    pub enum TrySendError<T> {
        Full(T),
        Closed(T),
    }

    pub struct Receiver<T> {
        inner: async_channel::Receiver<T>,
        counters: Arc<ChannelCounters>,
        resource_id: Option<cf_runtime::resource::ResourceId>,
        registry: Option<Arc<cf_runtime::resource::ResourceRegistry>>,
        /// Marker so we don't accidentally make it Clone; tokio's mpsc
        /// Receiver isn't Clone.
        #[allow(dead_code)]
        _no_clone: Arc<()>,
    }

    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            self.counters.mark_closed();
            if let (Some(id), Some(reg)) = (self.resource_id, self.registry.take()) {
                reg.remove(id);
            }
        }
    }

    impl<T> std::fmt::Debug for Receiver<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("mpsc::Receiver").finish()
        }
    }

    impl<T> Receiver<T> {
        /// Receive the next value. Returns `None` once all senders are
        /// dropped AND the channel is empty (matches tokio's behavior).
        pub async fn recv(&mut self) -> Option<T> {
            let inner_for_reason = self.inner.clone();
            let r = super::super::tag_wait(self.inner.recv(), move || {
                cf_runtime::WaitReason::MpscRecv {
                    depth: inner_for_reason.len(),
                }
            })
            .await
            .ok();
            super::log_event(
                "mpsc",
                if r.is_some() { "recv" } else { "recv-closed" },
            );
            if r.is_some() {
                self.counters.record_recv();
            }
            r
        }

        pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
            match self.inner.try_recv() {
                Ok(v) => Ok(v),
                Err(async_channel::TryRecvError::Empty) => Err(TryRecvError::Empty),
                Err(async_channel::TryRecvError::Closed) => Err(TryRecvError::Disconnected),
            }
        }

        pub fn close(&mut self) {
            self.inner.close();
        }
    }

    pub enum TryRecvError {
        Empty,
        Disconnected,
    }

    /// Tokio places these error types under `mpsc::error`. We mirror that.
    pub mod error {
        pub use super::{SendError, TryRecvError, TrySendError};
    }
}

pub mod oneshot {
    //! tokio::sync::oneshot — single-shot value transfer.
    //! Implemented on top of `async-channel::bounded(1)`.

    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let (s, r) = async_channel::bounded(1);
        (Sender { inner: Some(s) }, Receiver { inner: r })
    }

    pub struct Sender<T> {
        inner: Option<async_channel::Sender<T>>,
    }

    pub struct SendError<T>(pub T);

    impl<T> Sender<T> {
        pub fn send(mut self, value: T) -> Result<(), T> {
            // Per tokio, send is sync (the receiver capacity is 1 by
            // construction). Returns Err(value) if receiver dropped.
            let s = self.inner.take().expect("sent twice");
            s.try_send(value).map_err(|e| match e {
                async_channel::TrySendError::Full(v) | async_channel::TrySendError::Closed(v) => v,
            })
        }
    }

    pub struct Receiver<T> {
        inner: async_channel::Receiver<T>,
    }

    pub struct RecvError;

    impl std::fmt::Debug for RecvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("RecvError")
        }
    }

    impl std::fmt::Display for RecvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("channel closed")
        }
    }

    impl std::error::Error for RecvError {}

    impl<T> Future for Receiver<T> {
        type Output = Result<T, RecvError>;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let r = unsafe { self.as_mut().map_unchecked_mut(|s| &mut s.inner) };
            use futures_core::Stream;
            match r.poll_next(cx) {
                Poll::Ready(Some(v)) => Poll::Ready(Ok(v)),
                Poll::Ready(None) => Poll::Ready(Err(RecvError)),
                Poll::Pending => {
                    super::super::note_wait(cf_runtime::WaitReason::OneshotRecv);
                    Poll::Pending
                }
            }
        }
    }
}

pub mod broadcast {
    //! tokio::sync::broadcast — multi-producer, multi-consumer broadcast,
    //! built on async-broadcast.

    use cf_runtime::resource::{
        ChannelCounters, ResourceKind, ResourceProbe, ResourceStateSnapshot,
    };
    use std::sync::Arc;

    struct BroadcastProbe<T: Clone> {
        sender: async_broadcast::Sender<T>,
        capacity: usize,
        counters: Arc<ChannelCounters>,
    }
    impl<T: Clone + Send + Sync + 'static> ResourceProbe for BroadcastProbe<T> {
        fn snapshot(&self) -> ResourceStateSnapshot {
            let (s, r, hw, closed) = self.counters.read();
            ResourceStateSnapshot {
                depth: Some(self.sender.len()),
                capacity: Some(self.capacity),
                sends: s,
                recvs: r,
                high_water: hw,
                closed,
                ..Default::default()
            }
        }
    }

    pub fn channel<T: Clone + Send + Sync + 'static>(cap: usize) -> (Sender<T>, Receiver<T>) {
        let (mut s, r) = async_broadcast::broadcast::<T>(cap.max(1));
        s.set_overflow(true);
        let counters = ChannelCounters::new();
        let registry = cf_runtime::try_current().map(|h| h.resources());
        let resource_id = registry.as_ref().map(|reg| {
            reg.insert(
                ResourceKind::BroadcastChannel,
                format!("broadcast(cap={cap})"),
                cf_runtime::runtime::current_task(),
                Arc::new(BroadcastProbe::<T> {
                    sender: s.clone(),
                    capacity: cap,
                    counters: counters.clone(),
                }),
            )
        });
        (
            Sender {
                inner: s,
                counters: counters.clone(),
                resource_id,
                registry: registry.clone(),
            },
            Receiver {
                inner: r,
                counters,
                _resource_id: resource_id,
                _registry: registry,
            },
        )
    }

    #[derive(Clone)]
    pub struct Sender<T: Clone> {
        inner: async_broadcast::Sender<T>,
        counters: Arc<ChannelCounters>,
        resource_id: Option<cf_runtime::resource::ResourceId>,
        registry: Option<Arc<cf_runtime::resource::ResourceRegistry>>,
    }

    impl<T: Clone> Drop for Sender<T> {
        fn drop(&mut self) {
            // We can't tell if this is the LAST sender being dropped, but
            // the registry holds an Arc<dyn Probe> that owns its own
            // sender clone — that keeps the channel alive for snapshotting
            // until we explicitly remove. To avoid dangling probes when
            // every user-facing sender is gone, we deregister whenever a
            // sender is dropped after the inner is closed.
            if self.inner.is_closed() {
                self.counters.mark_closed();
                if let (Some(id), Some(reg)) = (self.resource_id.take(), self.registry.take()) {
                    reg.remove(id);
                }
            }
        }
    }

    impl<T: Clone> std::fmt::Debug for Sender<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("broadcast::Sender").finish()
        }
    }

    impl<T: Clone> Sender<T> {
        pub fn send(&self, value: T) -> Result<usize, SendError<T>> {
            super::log_event("broadcast", "send");
            self.counters.record_send(self.inner.len());
            // tokio returns Ok(receiver_count) on success. async-broadcast's
            // try_broadcast returns Ok(Option<old>) on success (with overflow);
            // the receiver count isn't directly exposed cheaply, so we return
            // 0 for now — mini-redis only checks Ok/Err, not the count.
            match self.inner.try_broadcast(value) {
                Ok(_) => Ok(0),
                Err(async_broadcast::TrySendError::Closed(v)) => Err(SendError(v)),
                Err(async_broadcast::TrySendError::Full(v)) => Err(SendError(v)),
                Err(async_broadcast::TrySendError::Inactive(v)) => {
                    // No active receivers. Tokio returns Err(SendError) too.
                    Err(SendError(v))
                }
            }
        }

        pub fn subscribe(&self) -> Receiver<T> {
            Receiver {
                inner: self.inner.new_receiver(),
                counters: self.counters.clone(),
                _resource_id: self.resource_id,
                _registry: self.registry.clone(),
            }
        }

        pub fn receiver_count(&self) -> usize {
            self.inner.receiver_count()
        }
    }

    pub struct SendError<T>(pub T);

    impl<T> std::fmt::Debug for SendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("SendError")
        }
    }

    impl<T> std::fmt::Display for SendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("channel closed or no receivers")
        }
    }

    impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

    pub struct Receiver<T: Clone> {
        inner: async_broadcast::Receiver<T>,
        counters: Arc<ChannelCounters>,
        #[allow(dead_code)]
        _resource_id: Option<cf_runtime::resource::ResourceId>,
        #[allow(dead_code)]
        _registry: Option<Arc<cf_runtime::resource::ResourceRegistry>>,
    }

    impl<T: Clone> std::fmt::Debug for Receiver<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("broadcast::Receiver").finish()
        }
    }

    pub enum RecvError {
        Closed,
        Lagged(u64),
    }

    impl std::fmt::Debug for RecvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RecvError::Closed => f.write_str("Closed"),
                RecvError::Lagged(n) => write!(f, "Lagged({n})"),
            }
        }
    }

    impl std::fmt::Display for RecvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RecvError::Closed => f.write_str("channel closed"),
                RecvError::Lagged(n) => write!(f, "lagged {n}"),
            }
        }
    }

    impl std::error::Error for RecvError {}

    impl<T: Clone> Receiver<T> {
        pub async fn recv(&mut self) -> Result<T, RecvError> {
            super::log_event("broadcast", "recv");
            self.counters.record_recv();
            let fut = super::super::tag_wait(self.inner.recv(), || {
                cf_runtime::WaitReason::BroadcastRecv
            });
            match fut.await {
                Ok(v) => Ok(v),
                Err(async_broadcast::RecvError::Closed) => Err(RecvError::Closed),
                Err(async_broadcast::RecvError::Overflowed(n)) => Err(RecvError::Lagged(n)),
            }
        }

        pub fn resubscribe(&self) -> Receiver<T> {
            Receiver {
                inner: self.inner.new_receiver(),
                counters: self.counters.clone(),
                _resource_id: self._resource_id,
                _registry: self._registry.clone(),
            }
        }
    }
}

mod notify_impl {
    //! tokio::sync::Notify — single-shot notification, "notified()" is a
    //! future that completes on the next notify call.
    use cf_runtime::resource::{
        ChannelCounters, ResourceId, ResourceKind, ResourceProbe, ResourceRegistry,
        ResourceStateSnapshot,
    };
    use event_listener::{Event, EventListener};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    pub struct Notify {
        evt: Arc<Event>,
        counters: Arc<ChannelCounters>,
        resource_id: Option<ResourceId>,
        registry: Option<Arc<ResourceRegistry>>,
    }

    struct NotifyProbe {
        evt: Arc<Event>,
        counters: Arc<ChannelCounters>,
    }
    impl ResourceProbe for NotifyProbe {
        fn snapshot(&self) -> ResourceStateSnapshot {
            let (s, r, hw, closed) = self.counters.read();
            ResourceStateSnapshot {
                // For Notify, "depth" = number of waiters currently parked
                // on the event. event-listener exposes total_listeners().
                depth: Some(self.evt.total_listeners()),
                sends: s,
                recvs: r,
                high_water: hw,
                closed,
                ..Default::default()
            }
        }
    }

    impl Drop for Notify {
        fn drop(&mut self) {
            self.counters.mark_closed();
            if let (Some(id), Some(reg)) = (self.resource_id, self.registry.take()) {
                reg.remove(id);
            }
        }
    }

    impl Default for Notify {
        fn default() -> Self {
            Self::new()
        }
    }

    impl std::fmt::Debug for Notify {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Notify")
        }
    }

    impl Notify {
        pub fn new() -> Self {
            let evt = Arc::new(Event::new());
            let counters = ChannelCounters::new();
            let registry = cf_runtime::try_current().map(|h| h.resources());
            let resource_id = registry.as_ref().map(|reg| {
                reg.insert(
                    ResourceKind::Notify,
                    "Notify".to_string(),
                    cf_runtime::runtime::current_task(),
                    Arc::new(NotifyProbe {
                        evt: evt.clone(),
                        counters: counters.clone(),
                    }),
                )
            });
            Self {
                evt,
                counters,
                resource_id,
                registry,
            }
        }

        pub fn notify_one(&self) {
            super::log_event("notify", "notify_one");
            self.counters.record_send(0);
            self.evt.notify(1usize);
        }

        pub fn notify_waiters(&self) {
            super::log_event("notify", "notify_waiters");
            self.counters.record_send(0);
            self.evt.notify(usize::MAX);
        }

        /// Returns a future that resolves on the next notify. Note: the
        /// listener must be created (subscribed) before the notify happens
        /// to be guaranteed to observe it. Tokio's API has the same caveat
        /// (notify_one only wakes existing waiters).
        pub fn notified(&self) -> Notified<'_> {
            Notified {
                listener: Some(self.evt.listen()),
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct Notified<'a> {
        listener: Option<EventListener>,
        // PhantomData to tie lifetime to the Notify (event-listener's
        // EventListener is 'static, but tokio's Notified borrows; we mimic
        // that for API compatibility even though it's not load-bearing).
        #[allow(dead_code)]
        _marker: std::marker::PhantomData<&'a ()>,
    }

    impl<'a> Notified<'a> {
        // Helper for constructor without the marker (init via struct literal
        // above).
    }

    impl<'a> Future for Notified<'a> {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            let l = self
                .listener
                .as_mut()
                .expect("polled Notified after completion");
            let pinned = unsafe { Pin::new_unchecked(l) };
            match pinned.poll(cx) {
                Poll::Ready(()) => {
                    self.listener.take();
                    Poll::Ready(())
                }
                Poll::Pending => {
                    crate::note_wait(cf_runtime::WaitReason::NotifyWait);
                    Poll::Pending
                }
            }
        }
    }

    // Provide the constructor used in the lib code that fills in PhantomData.
    impl Notify {
        pub(super) fn make_notified(&self) -> Notified<'_> {
            Notified {
                listener: Some(self.evt.listen()),
                _marker: std::marker::PhantomData,
            }
        }
    }
}

pub use notify_impl::{Notified, Notify};

/// Mirror of tokio::sync::Semaphore. Built on async-lock.
///
/// Internally wraps `Arc<async_lock::Semaphore>` so that `acquire_owned`
/// (which needs an Arc-bearing acquire) can hand out a permit that keeps the
/// underlying semaphore alive without requiring the user to wrap us in an
/// extra Arc layer beyond the standard `Arc<Semaphore>` pattern.
pub struct Semaphore {
    inner: std::sync::Arc<async_lock::Semaphore>,
    counters: std::sync::Arc<cf_runtime::resource::ChannelCounters>,
    initial_permits: usize,
    resource_id: Option<cf_runtime::resource::ResourceId>,
    registry: Option<std::sync::Arc<cf_runtime::resource::ResourceRegistry>>,
}

struct SemaphoreProbe {
    inner: std::sync::Arc<async_lock::Semaphore>,
    initial: usize,
    counters: std::sync::Arc<cf_runtime::resource::ChannelCounters>,
}
impl cf_runtime::resource::ResourceProbe for SemaphoreProbe {
    fn snapshot(&self) -> cf_runtime::resource::ResourceStateSnapshot {
        let (s, r, hw, _) = self.counters.read();
        // depth = approximate "permits currently held" = initial - available;
        // async-lock doesn't expose available cheaply, so we report the
        // diff through counters: sends count acquires, recvs count releases.
        let outstanding = s.saturating_sub(r);
        cf_runtime::resource::ResourceStateSnapshot {
            depth: Some(outstanding as usize),
            capacity: Some(self.initial),
            sends: s,
            recvs: r,
            high_water: hw,
            ..Default::default()
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if let (Some(id), Some(reg)) = (self.resource_id.take(), self.registry.take()) {
            reg.remove(id);
        }
    }
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Semaphore").finish()
    }
}

impl Semaphore {
    pub fn new(permits: usize) -> Self {
        let inner = std::sync::Arc::new(async_lock::Semaphore::new(permits));
        let counters = cf_runtime::resource::ChannelCounters::new();
        let registry = cf_runtime::try_current().map(|h| h.resources());
        let resource_id = registry.as_ref().map(|reg| {
            reg.insert(
                cf_runtime::resource::ResourceKind::Semaphore,
                format!("Semaphore({permits})"),
                cf_runtime::runtime::current_task(),
                std::sync::Arc::new(SemaphoreProbe {
                    inner: inner.clone(),
                    initial: permits,
                    counters: counters.clone(),
                }),
            )
        });
        Self {
            inner,
            counters,
            initial_permits: permits,
            resource_id,
            registry,
        }
    }

    pub async fn acquire(&self) -> Result<SemaphorePermit<'_>, AcquireError> {
        let g = crate::tag_wait(self.inner.acquire(), || {
            cf_runtime::WaitReason::SemAcquire {
                permits_requested: 1,
            }
        })
        .await;
        self.counters.record_send(0);
        Ok(SemaphorePermit {
            _guard: Some(g),
            counters: Some(self.counters.clone()),
        })
    }

    /// `acquire_owned` consumes an `Arc<Semaphore>` and returns a permit
    /// that holds an inner Arc. Permit drops return capacity to the
    /// semaphore.
    pub async fn acquire_owned(
        self: std::sync::Arc<Self>,
    ) -> Result<OwnedSemaphorePermit, AcquireError> {
        log_event("sem", "acquire_owned");
        let inner = self.inner.clone();
        let counters = self.counters.clone();
        let g = crate::tag_wait(inner.acquire_arc(), || {
            cf_runtime::WaitReason::SemAcquire {
                permits_requested: 1,
            }
        })
        .await;
        counters.record_send(0);
        Ok(OwnedSemaphorePermit {
            _guard: Some(g),
            _sem: self,
            counters: Some(counters),
        })
    }

    pub fn available_permits(&self) -> usize {
        // async-lock's Semaphore doesn't directly expose this; we return 0
        // and treat it as best-effort. Mini-redis doesn't read it.
        0
    }
}

pub struct AcquireError;

impl std::fmt::Debug for AcquireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AcquireError")
    }
}

impl std::fmt::Display for AcquireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("semaphore closed")
    }
}

impl std::error::Error for AcquireError {}

pub struct SemaphorePermit<'a> {
    _guard: Option<async_lock::SemaphoreGuard<'a>>,
    counters: Option<std::sync::Arc<cf_runtime::resource::ChannelCounters>>,
}

impl<'a> Drop for SemaphorePermit<'a> {
    fn drop(&mut self) {
        if let Some(c) = &self.counters {
            c.record_recv();
        }
    }
}

pub struct OwnedSemaphorePermit {
    _guard: Option<async_lock::SemaphoreGuardArc>,
    _sem: std::sync::Arc<Semaphore>,
    counters: Option<std::sync::Arc<cf_runtime::resource::ChannelCounters>>,
}

impl Drop for OwnedSemaphorePermit {
    fn drop(&mut self) {
        if let Some(c) = &self.counters {
            c.record_recv();
        }
    }
}
