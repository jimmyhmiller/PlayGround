//! Event queue. Thread-safe so generator threads can push events while the
//! main scheduler thread pops.
//!
//! Exit semantics: `pop_blocking` returns `None` only when the queue is
//! empty AND there are no active generator threads. Tests that don't use
//! generators still see the queue drain and exit normally (they register
//! zero generators).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use crate::value::Value;

#[derive(Debug, Clone)]
pub struct InboundEvent {
    pub event: String,
    pub payload: Value,
    /// Optional correlation: set by the runtime when an effect response is
    /// converted back into an event.
    pub correlates_to: Option<u64>,
}

impl InboundEvent {
    pub fn new(event: impl Into<String>, payload: Value) -> Self {
        Self {
            event: event.into(),
            payload,
            correlates_to: None,
        }
    }
}

#[derive(Default)]
struct Inner {
    queue: VecDeque<InboundEvent>,
    /// Generator threads that have not yet finished. The scheduler keeps
    /// blocking as long as this is > 0 even when the queue is empty.
    generators_running: usize,
}

#[derive(Clone, Default)]
pub struct EventQueue {
    inner: Arc<(Mutex<Inner>, Condvar)>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an event. Wakes the scheduler if it was waiting.
    pub fn push_back(&self, e: InboundEvent) {
        let (lock, cvar) = &*self.inner;
        let mut g = lock.lock().unwrap();
        g.queue.push_back(e);
        cvar.notify_one();
    }

    /// Pop the next event, blocking until one is available. Returns `None`
    /// only when the queue is empty AND no generators are still running.
    pub fn pop_blocking(&self) -> Option<InboundEvent> {
        let (lock, cvar) = &*self.inner;
        let mut g = lock.lock().unwrap();
        loop {
            if let Some(ev) = g.queue.pop_front() {
                return Some(ev);
            }
            if g.generators_running == 0 {
                return None;
            }
            g = cvar.wait(g).unwrap();
        }
    }

    /// Like `pop_blocking` but non-blocking. Used by tests / run_to_quiescence
    /// where blocking on absent generators isn't wanted.
    pub fn pop_front(&self) -> Option<InboundEvent> {
        let (lock, _) = &*self.inner;
        lock.lock().unwrap().queue.pop_front()
    }

    pub fn len(&self) -> usize {
        let (lock, _) = &*self.inner;
        lock.lock().unwrap().queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mark a new generator as running. Called when the runtime starts a
    /// generator thread.
    pub fn generator_started(&self) {
        let (lock, _) = &*self.inner;
        lock.lock().unwrap().generators_running += 1;
    }

    /// Mark a generator as finished. Called when its thread exits.
    pub fn generator_finished(&self) {
        let (lock, cvar) = &*self.inner;
        let mut g = lock.lock().unwrap();
        g.generators_running = g.generators_running.saturating_sub(1);
        if g.generators_running == 0 {
            cvar.notify_all();
        }
    }
}
