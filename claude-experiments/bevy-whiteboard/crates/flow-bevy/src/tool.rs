//! Active-tool state — what mode the whiteboard is in.

use bevy::prelude::*;

pub struct ToolPlugin;
impl Plugin for ToolPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveTool>()
            .init_resource::<ActiveSlot>()
            .init_resource::<NodeColors>();
    }
}

/// Snapshot of each flow node's data-palette colour at drop time, keyed by
/// sim `NodeId`. Lives on the Bevy side — the flow sim doesn't know (or
/// care) about colour. Edge packet visuals look this up to tint themselves
/// with the emitter's colour; `None` means fall back to `theme.accent`.
#[derive(Resource, Default)]
pub struct NodeColors(pub std::collections::HashMap<flow::NodeId, bevy::prelude::Color>);

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    #[default]
    Select,
    /// Drop a gadget at the next click. Carries which kind.
    Drop(crate::gadgets::Kind),
    /// Connecting edge: first click picks source, second picks target.
    Connect,
    /// Attach a probe to the next clicked edge.
    Probe,
}

#[derive(Resource, Default)]
pub struct ActiveTool(pub Tool);

/// Which data-palette slot is currently selected. Indexes into `Theme::data`
/// — tracked as an index (not a `Color` value) so a theme swap doesn't leave
/// the user on a stale colour. Nodes dropped while this slot is active get
/// tagged with `theme.data[slot]` at drop time (snapshot, so the colour
/// survives subsequent theme changes).
#[derive(Resource)]
pub struct ActiveSlot(pub usize);

impl Default for ActiveSlot {
    fn default() -> Self { Self(0) }
}
