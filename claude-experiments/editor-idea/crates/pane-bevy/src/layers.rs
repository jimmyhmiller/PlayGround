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
///
/// `reserved` holds layer ids that belong to something OTHER than a pane
/// and must never be handed out. The host passes these in (see
/// [`with_reserved`](Self::with_reserved)) — e.g. the menu-overlay
/// camera's layer. That camera renders its layer at a very high order
/// and is NOT project-scoped, so if a pane were ever allocated the same
/// id, the overlay camera would draw that pane's content globally: over
/// menus (pin menu "behind" the pane) and across every project (a pane
/// "shadowing" onto others). Reserving makes that collision impossible
/// rather than merely unlikely-until-you-have-32-panes.
#[derive(Resource, Default)]
pub struct PaneLayerAllocator {
    next: usize,
    /// Returned-but-not-yet-reused ids. `BTreeSet` for deterministic
    /// reuse (smallest first), which keeps debug output stable.
    free: BTreeSet<usize>,
    /// Layer ids that are never allocated (owned by non-pane cameras).
    reserved: BTreeSet<usize>,
}

impl PaneLayerAllocator {
    pub fn new() -> Self {
        Self::with_reserved(std::iter::empty())
    }

    /// Allocator that never hands out any id in `reserved` (nor `0`).
    pub fn with_reserved(reserved: impl IntoIterator<Item = usize>) -> Self {
        Self {
            next: 1,
            free: BTreeSet::new(),
            reserved: reserved.into_iter().filter(|&id| id != 0).collect(),
        }
    }

    pub fn allocate(&mut self) -> usize {
        // Reuse a freed id (smallest first), skipping any reserved.
        while let Some(&id) = self.free.iter().next() {
            self.free.remove(&id);
            if !self.reserved.contains(&id) {
                return id;
            }
        }
        // Otherwise bump the counter, stepping over reserved ids so a
        // pane is NEVER assigned a reserved layer.
        loop {
            let id = self.next;
            self.next += 1;
            if !self.reserved.contains(&id) {
                return id;
            }
        }
    }

    /// Return an id to the pool. No-op for `0` or reserved ids (we never
    /// hand those out, so they can't come back).
    pub fn free(&mut self, id: usize) {
        if id == 0 || self.reserved.contains(&id) {
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

    #[test]
    fn reserved_layers_are_never_allocated() {
        // Reserve 2 and 4; the counter must step over them.
        let mut a = PaneLayerAllocator::with_reserved([2, 4]);
        assert_eq!(a.allocate(), 1);
        assert_eq!(a.allocate(), 3); // 2 skipped
        assert_eq!(a.allocate(), 5); // 4 skipped
        assert_eq!(a.allocate(), 6);
    }

    #[test]
    fn the_menu_overlay_layer_collision_cannot_happen() {
        // Regression: 40 panes with layer 32 reserved must never produce
        // 32 (the menu-overlay layer), which previously leaked panes onto
        // the overlay camera once >31 panes existed.
        let mut a = PaneLayerAllocator::with_reserved([32]);
        for _ in 0..40 {
            assert_ne!(a.allocate(), 32);
        }
    }
}
