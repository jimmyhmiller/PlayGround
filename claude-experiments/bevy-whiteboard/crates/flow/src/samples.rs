use std::collections::VecDeque;

use crate::value::Value;

/// A bounded ring of recent samples. Push appends; when full, the
/// oldest sample is evicted. Supports pop-oldest, stats, and
/// predicate-based drop.
///
/// This is the closest thing to a collection slot-type we ship. It's
/// *bounded*, so modeling "10M outstanding requests" is still a number
/// (count slot) — the ring only keeps enough recent arrival-times to
/// estimate age distributions faithfully.
#[derive(Debug, Clone)]
pub struct Samples {
    pub cap: usize,
    pub items: VecDeque<Value>,
}

impl Samples {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "Samples cap must be > 0");
        Self { cap, items: VecDeque::with_capacity(cap) }
    }


    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }

    pub fn push(&mut self, v: Value) {
        if self.items.len() == self.cap {
            self.items.pop_front();
        }
        self.items.push_back(v);
    }

    pub fn pop_oldest(&mut self) -> Option<Value> {
        self.items.pop_front()
    }

    /// Mean of the samples, interpreted as f64. Panics if any sample
    /// isn't Int or Float — that's a modeling error, make it loud.
    pub fn mean_f64(&self) -> f64 {
        if self.items.is_empty() { return 0.0; }
        let sum: f64 = self.items.iter().map(|v| match v {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            other => panic!("Samples::mean_f64: non-numeric sample {:?}", other),
        }).sum();
        sum / (self.items.len() as f64)
    }
}
