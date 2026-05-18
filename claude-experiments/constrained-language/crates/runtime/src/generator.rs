//! Generators: typed sources of events.
//!
//! A generator is the dual of an effect adapter: where an `Adapter` is a
//! function the runtime calls to *consume* an effect request and return a
//! response, a `Generator` is a function the runtime calls to *produce* the
//! next payload for a declared event. Each generator runs on a dedicated
//! thread; results are pushed onto the runtime queue. When a generator
//! returns `None`, the runtime marks it finished; when all generators
//! finish and the queue drains, `run_until_idle` returns.
//!
//! v0.1 supports the pull shape only: `next() -> Option<payload>`. Push
//! generators (those that own their own internal loop and call into the
//! runtime when ready) will arrive as a second trait once we have a real
//! use case.

use std::collections::HashMap;

use crate::value::Value;

/// Pull-style event source. The runtime calls `next` in a loop on a
/// dedicated thread; each `Some(payload)` becomes an event matching the
/// generator's declared `event`. `None` ends the stream.
pub trait Generator: Send {
    fn next(&mut self) -> Option<Value>;
}

#[derive(Default)]
pub struct GeneratorRegistry {
    generators: HashMap<String, Box<dyn Generator>>,
}

impl GeneratorRegistry {
    pub fn register(&mut self, name: impl Into<String>, generator: Box<dyn Generator>) {
        self.generators.insert(name.into(), generator);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.generators.contains_key(name)
    }

    /// Remove a generator by name. Used by the runtime when starting a
    /// generator's thread (the generator owns itself from then on).
    pub fn take(&mut self, name: &str) -> Option<Box<dyn Generator>> {
        self.generators.remove(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.generators.keys().cloned().collect()
    }
}
