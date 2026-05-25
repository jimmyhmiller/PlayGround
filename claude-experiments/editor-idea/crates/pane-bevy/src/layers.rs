//! Per-pane RenderLayers allocation.
//!
//! Each pane owns a unique `usize` layer id (1, 2, 3, …). Layer `0` is
//! reserved for the main camera + pane chrome. The id is the index
//! into Bevy's `RenderLayers` bitset. Later stages will:
//!
//! - Spawn a per-pane Camera2d filtered to the pane's layer.
//! - Propagate `RenderLayers::layer(id)` to everything under the
//!   pane's content_root, so its descendants are visible only to the
//!   pane camera and clipped to that camera's viewport.
//!
//! This module owns the id lifecycle. The `PaneLayer` component is
//! inert until later stages read it; nothing here changes what
//! renders.
//!
//! Freeing: Bevy's `RemovedComponents` gives entities but not their
//! prior component values, so we can't read the id at removal time
//! through that API. Instead the close flow that owns pane teardown
//! reads the `PaneLayer` value off the entity before despawn and
//! calls `PaneLayerAllocator::free` directly. See `Stage 2` for the
//! actual wire-up.

use bevy::prelude::*;
use std::collections::BTreeSet;

/// Stable identity for which RenderLayer a pane owns. Inserted on the
/// pane entity at spawn time.
#[derive(Component, Copy, Clone, Debug, PartialEq, Eq)]
pub struct PaneLayer(pub usize);

/// Hands out unique RenderLayer ids and reclaims them when callers
/// explicitly free. Layer `0` is reserved (main camera + chrome), so
/// allocations begin at `1`. Freed ids are reused before bumping the
/// counter so the live set stays dense.
#[derive(Resource, Default)]
pub struct PaneLayerAllocator {
    next: usize,
    /// Returned-but-not-yet-reused ids. `BTreeSet` for deterministic
    /// reuse (smallest first), which keeps debug output stable.
    free: BTreeSet<usize>,
}

impl PaneLayerAllocator {
    pub fn new() -> Self {
        Self {
            next: 1,
            free: BTreeSet::new(),
        }
    }

    pub fn allocate(&mut self) -> usize {
        if let Some(&id) = self.free.iter().next() {
            self.free.remove(&id);
            return id;
        }
        let id = self.next;
        self.next += 1;
        id
    }

    /// Return an id to the pool. No-op if `id == 0` since we never
    /// hand out 0.
    pub fn free(&mut self, id: usize) {
        if id == 0 {
            return;
        }
        self.free.insert(id);
    }

    #[allow(dead_code)]
    pub fn allocated_count(&self) -> usize {
        self.next.saturating_sub(1).saturating_sub(self.free.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_starts_at_one() {
        let mut a = PaneLayerAllocator::new();
        assert_eq!(a.allocate(), 1);
        assert_eq!(a.allocate(), 2);
        assert_eq!(a.allocate(), 3);
    }

    #[test]
    fn freed_ids_reused_smallest_first() {
        let mut a = PaneLayerAllocator::new();
        let _ = a.allocate(); // 1
        let b = a.allocate(); // 2
        let c = a.allocate(); // 3
        a.free(b);
        a.free(c);
        assert_eq!(a.allocate(), 2);
        assert_eq!(a.allocate(), 3);
        assert_eq!(a.allocate(), 4);
    }

    #[test]
    fn cannot_free_zero() {
        let mut a = PaneLayerAllocator::new();
        a.free(0);
        assert_eq!(a.allocate(), 1);
    }
}
