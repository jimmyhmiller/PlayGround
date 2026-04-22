//! `poster-ui` — a Bevy UI toolkit for iso50 / mid-century-poster interfaces.
//!
//! The toolkit is intentionally opinionated about visual direction (cream
//! paper, ink/accent palette, tracked-caps typography, bordered panels with
//! rounded corners) but knows nothing about your application's domain. Bring
//! your own Tool enum, sim, node graph — wire them into the builders and the
//! re-skin systems will keep the chrome looking right when the theme swaps.
//!
//! Typical `main.rs`:
//! ```ignore
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(poster_ui::PosterUiPlugin)
//!     .add_systems(Startup, spawn_my_palette)
//!     .run();
//! ```
//!
//! Then inside a spawn system you compose the app's panel from the provided
//! primitives (see [`panel`]) and its bottom HUD from the cell builders
//! (see [`hud`]).

use bevy::prelude::*;

pub mod hud;
pub mod panel;
pub mod scroll;
pub mod slider;
pub mod theme;
#[cfg(feature = "testing")]
pub mod testing;
pub mod typography;

pub use hud::*;
pub use panel::*;
pub use scroll::*;
pub use slider::*;
pub use theme::*;
pub use typography::*;

/// Bundle plugin: installs the theme resource, typography (font loading +
/// Bold/Mono stamping), and mousewheel-driven scroll pane routing.
///
/// If you want to change the starting theme, add
/// `.insert_resource(poster_ui::Theme::dark())` *after* adding this plugin.
pub struct PosterUiPlugin;

impl Plugin for PosterUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            theme::ThemePlugin,
            typography::TypographyPlugin,
            scroll::ScrollPlugin,
            panel::PanelReskinPlugin,
            hud::HudReskinPlugin,
            slider::SliderPlugin,
        ));
    }
}
