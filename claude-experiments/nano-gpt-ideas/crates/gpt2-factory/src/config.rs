//! Architectural constants for the factory layout.
//!
//! Units are "meters" (unitless but eye-height ~1.7). The tower is
//! generated procedurally from these so everything stays consistent.

pub const N_LAYER: usize = 12;

/// Height of one transformer block (attention hall + mlp chamber stacked).
pub const BLOCK_HEIGHT: f32 = 12.0;
/// Vertical gap between blocks (a visible seam where the residual belt
/// passes through a thin floor slab).
pub const BLOCK_GAP: f32 = 1.0;

/// Attention hall occupies the bottom 2/3 of a block; MLP the top 1/3.
pub const ATTN_HEIGHT: f32 = 8.0;
pub const MLP_HEIGHT: f32 = 4.0;

/// X/Z footprint of a block room.
pub const BLOCK_WIDTH: f32 = 40.0;
pub const BLOCK_DEPTH: f32 = 24.0;

/// Wall thickness / slab thickness.
pub const SLAB_THICKNESS: f32 = 0.3;

/// How far the entry and exit halls extend past the ends of the tower.
pub const HALL_LENGTH: f32 = 30.0;

/// Total height of the tower body (no halls).
pub fn tower_total_height() -> f32 {
    N_LAYER as f32 * (BLOCK_HEIGHT + BLOCK_GAP) + BLOCK_GAP
}

/// Y-coordinate of floor i's floor slab (i = 0..=N_LAYER).
pub fn floor_y(i: usize) -> f32 {
    i as f32 * (BLOCK_HEIGHT + BLOCK_GAP)
}

/// Y-coordinate of the center of the attention hall on layer `layer`.
pub fn attn_center_y(layer: usize) -> f32 {
    floor_y(layer) + BLOCK_GAP + ATTN_HEIGHT * 0.5
}

/// Y-coordinate of the center of the mlp chamber on layer `layer`.
pub fn mlp_center_y(layer: usize) -> f32 {
    floor_y(layer) + BLOCK_GAP + ATTN_HEIGHT + MLP_HEIGHT * 0.5
}
