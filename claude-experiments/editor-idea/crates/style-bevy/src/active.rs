//! `ActiveProject` — the host tells us which project is currently
//! active; the dynamic-shader pipeline scopes its per-project masks
//! to this id. That's the only piece of data that needs to flow from
//! the host into style-bevy at runtime (everything else either lives
//! on disk or comes from Bevy resources scripts read directly).
//!
//! Its own tiny module so removing the old shader.rs/wipe.rs/dev.rs
//! doesn't take it with them.

use bevy::prelude::*;

#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct ActiveProject(pub Option<u64>);
