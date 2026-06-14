//! Input handling and the tool state machine.
//!
//! This module turns raw pointer / keyboard / wheel events into scene mutations:
//! drag-to-create, select, marquee, move, resize (8 handles), rotate, pan, and
//! zoom. The headless library owns *all* interaction — a backend only forwards
//! raw events and renders the result.
//!
//! Phase 1 implements the per-tool behavior and selection/handle logic. This
//! file defines the shared input vocabulary ([`InputEvent`], [`Tool`],
//! modifiers, pointer buttons) plus the [`Viewport`] (pan/zoom) that every tool
//! and the renderer consult.

use crate::geometry::{Point, Transform, Vec2};
use serde::{Deserialize, Serialize};

/// The active drawing/selection tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Tool {
    #[default]
    Select,
    Pan,
    Rectangle,
    Ellipse,
    Diamond,
    Line,
    Arrow,
    Freedraw,
    Text,
    Image,
    Frame,
    Eraser,
    Laser,
}

/// Mouse / pointer button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointerButton {
    Primary,
    Secondary,
    Middle,
}

/// Keyboard modifier state at the moment of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    /// Cmd on macOS, Super/Win elsewhere.
    pub meta: bool,
}

impl Modifiers {
    /// The platform "command" modifier (ctrl on most platforms, meta on macOS).
    /// Callers normalize per platform; we treat either as "command" here.
    pub fn command(&self) -> bool {
        self.ctrl || self.meta
    }
}

/// A keyboard key, abstracted away from physical scan codes. Phase 1 grows this
/// as shortcuts are implemented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Key {
    Escape,
    Enter,
    Backspace,
    Delete,
    Tab,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    /// A printable character.
    Char(char),
    /// Anything not otherwise modeled, by name.
    Named(String),
}

/// A raw input event handed to the editor. Positions are in **screen**
/// coordinates; the editor maps them to scene coordinates via the [`Viewport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputEvent {
    PointerDown {
        pos: Point,
        button: PointerButton,
        mods: Modifiers,
    },
    PointerMove {
        pos: Point,
        mods: Modifiers,
    },
    PointerUp {
        pos: Point,
        button: PointerButton,
        mods: Modifiers,
    },
    KeyDown {
        key: Key,
        mods: Modifiers,
    },
    KeyUp {
        key: Key,
        mods: Modifiers,
    },
    /// Wheel / trackpad scroll. `delta` is in screen units; `mods` decides pan
    /// vs zoom per platform convention (ctrl/meta ⇒ zoom).
    Wheel {
        delta: Vec2,
        pos: Point,
        mods: Modifiers,
    },
}

/// Pan/zoom state mapping between screen and scene coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    /// Scene-space point shown at the screen origin.
    pub scroll: Vec2,
    /// Scale factor (1.0 = 100%).
    pub zoom: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Viewport {
            scroll: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl Viewport {
    /// Transform mapping scene coordinates to screen coordinates.
    pub fn scene_to_screen(&self) -> Transform {
        Transform::translate(-self.scroll.x, -self.scroll.y)
            .then(&Transform::scale(self.zoom, self.zoom))
    }

    /// Map a screen point into scene coordinates.
    pub fn screen_to_scene(&self, p: Point) -> Point {
        Point::new(
            p.x / self.zoom + self.scroll.x,
            p.y / self.zoom + self.scroll.y,
        )
    }

    /// Zoom toward a fixed screen anchor (keeps the scene point under the cursor
    /// stationary while zooming). Clamps zoom to a sane range.
    pub fn zoom_to(&mut self, new_zoom: f64, anchor_screen: Point) {
        let new_zoom = new_zoom.clamp(0.05, 30.0);
        let anchor_scene = self.screen_to_scene(anchor_screen);
        self.zoom = new_zoom;
        // Re-derive scroll so anchor_scene maps back to anchor_screen.
        self.scroll = Vec2::new(
            anchor_scene.x - anchor_screen.x / new_zoom,
            anchor_scene.y - anchor_screen.y / new_zoom,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_scene_round_trip() {
        let vp = Viewport {
            scroll: Vec2::new(100.0, 50.0),
            zoom: 2.0,
        };
        let screen = Point::new(640.0, 480.0);
        let scene = vp.screen_to_scene(screen);
        let back = vp.scene_to_screen().apply(scene);
        assert!((back.x - screen.x).abs() < 1e-6, "x={}", back.x);
        assert!((back.y - screen.y).abs() < 1e-6, "y={}", back.y);
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut vp = Viewport::default();
        let anchor = Point::new(200.0, 200.0);
        let scene_before = vp.screen_to_scene(anchor);
        vp.zoom_to(3.0, anchor);
        let scene_after = vp.screen_to_scene(anchor);
        assert!((scene_before.x - scene_after.x).abs() < 1e-6);
        assert!((scene_before.y - scene_after.y).abs() < 1e-6);
    }

    #[test]
    fn command_modifier() {
        let m = Modifiers {
            ctrl: true,
            ..Default::default()
        };
        assert!(m.command());
    }
}
