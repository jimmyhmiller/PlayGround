//! Thin wrapper: the real simulation lives in `src/sim/`. The Bevy integration
//! (entity ↔ sim id maps, event-driven packet animation) lives in `src/bridge.rs`.
//! This plugin exists only so the existing `main.rs` plugin list keeps working
//! — it delegates to `BridgePlugin`.

use crate::bridge::BridgePlugin;
use bevy::prelude::*;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BridgePlugin);
    }
}
