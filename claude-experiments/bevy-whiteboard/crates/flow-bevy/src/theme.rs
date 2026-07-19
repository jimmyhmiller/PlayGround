//! Theme is owned by `poster-ui` now. This module re-exports it plus a small
//! helper mapping each flow `Kind` to its theme fill colour so call-sites
//! don't need to care about the `NodeFillSet` field names.
//!
//! If you want a different starting theme, add
//! `.insert_resource(poster_ui::Theme::dark())` after `PosterUiPlugin` in
//! `main.rs`.

pub use poster_ui::Theme;

use bevy::prelude::Color;

use crate::gadgets::Kind;

/// Look up the theme fill colour for a flow node kind. Delegates to the
/// canonical [`poster_ui::NodeFillSet`] slots.
pub fn kind_color(theme: &Theme, kind: Kind) -> Color {
    match kind {
        Kind::Generator => theme.node_fill.generator,
        Kind::Client => theme.node_fill.client,
        Kind::BackoffClient => theme.node_fill.client,
        Kind::Worker => theme.node_fill.worker,
        Kind::Router => theme.node_fill.router,
        Kind::Queue => theme.node_fill.queue,
        Kind::Sink => theme.node_fill.sink,
        // A fleet of workers — share the worker fill so the family reads
        // as related.
        Kind::AutoScalingGroup => theme.node_fill.worker,
    }
}
