//! The element model — the shared foundation every other module builds on.
//!
//! Reimplemented from Excalidraw's element schema
//! (`packages/element/src/types.ts`). We keep field names and semantics close to
//! upstream so `.excalidraw` files load losslessly, but use idiomatic Rust:
//! a base struct of common fields plus a `kind`-specific payload enum.

mod binding;
mod id;
mod kind;

pub use binding::{
    apply_bound_endpoints, bindable_element_at, bound_point, compute_binding, is_bindable_element,
    update_bound_arrow, BoundEndpoints,
};
pub use id::{ElementId, GroupId};
pub use kind::{
    Arrowhead, BoundElement, BoundElementKind, ElementKind, FrameData, FreedrawData, ImageData,
    ImageStatus, LinearData, LinearKind, PointBinding, Roundness, RoundnessKind, TextData,
};

use crate::geometry::{Point, Rect};
use crate::render::{Color, FillStyle, StrokeStyle};
use serde::{Deserialize, Serialize};

/// Fields common to every element, mirroring Excalidraw's `_ExcalidrawElementBase`.
///
/// `x`/`y` are the top-left of the element's *unrotated* bounding box in scene
/// coordinates; `angle` rotates the element clockwise around the box center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: ElementId,

    // Position & size (unrotated box).
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// Rotation in radians, clockwise, about the box center.
    pub angle: f64,

    // Styling.
    pub stroke_color: Color,
    pub background_color: Color,
    pub fill_style: FillStyle,
    pub stroke_width: f64,
    pub stroke_style: StrokeStyle,
    /// 0 = precise, higher = more hand-drawn jitter (Excalidraw uses 0/1/2).
    pub roughness: f64,
    /// 0.0..=100.0 (Excalidraw stores opacity as a percentage).
    pub opacity: f64,

    // Structure.
    /// Ordered group memberships, outermost last (matches Excalidraw).
    pub group_ids: Vec<GroupId>,
    /// Elements bound to this one (e.g. bound text, arrows pointing at it).
    pub bound_elements: Vec<BoundElement>,
    /// The frame this element belongs to, if any.
    pub frame_id: Option<ElementId>,

    // State.
    pub is_deleted: bool,
    pub locked: bool,
    /// Fractional index for stable z-ordering under concurrent edits.
    pub index: Option<String>,
    /// Random seed driving the hand-drawn shape generation (so a shape looks
    /// stable across renders).
    pub seed: u32,
    /// Bumped on every mutation; used for bounds caching and sync.
    pub version: u64,
    pub version_nonce: u32,
    /// Last-updated timestamp (ms). Optional; not all sources provide it.
    pub updated: Option<u64>,

    // Metadata.
    pub link: Option<String>,

    /// The element-type-specific payload.
    pub kind: ElementKind,
}

impl Element {
    /// Create an element of `kind` with the given box and sensible defaults.
    /// `id`/`seed` are caller-provided so creation stays deterministic and
    /// free of hidden global state (important for headless + tests).
    pub fn new(
        id: ElementId,
        seed: u32,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        kind: ElementKind,
    ) -> Self {
        Element {
            id,
            x,
            y,
            width,
            height,
            angle: 0.0,
            stroke_color: Color::rgb(30, 30, 30),
            background_color: Color::TRANSPARENT,
            fill_style: FillStyle::Hachure,
            stroke_width: 1.0,
            stroke_style: StrokeStyle::Solid,
            roughness: 1.0,
            opacity: 100.0,
            group_ids: Vec::new(),
            bound_elements: Vec::new(),
            frame_id: None,
            is_deleted: false,
            locked: false,
            index: None,
            seed,
            version: 1,
            version_nonce: 0,
            updated: None,
            link: None,
            kind,
        }
    }

    /// Opacity normalized to 0.0..=1.0 for rendering.
    pub fn opacity_unit(&self) -> f32 {
        (self.opacity / 100.0).clamp(0.0, 1.0) as f32
    }

    /// Center of the unrotated bounding box (also the rotation pivot).
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// The unrotated, axis-aligned bounding box from `x/y/width/height`.
    ///
    /// This is the raw box only. Tight bounds that account for rotation, curve
    /// extents, arrowheads and bound text are computed in the `geometry::bounds`
    /// module (Phase 1) — that is the function callers should normally use.
    pub fn raw_box(&self) -> Rect {
        Rect::new(self.x, self.y, self.width, self.height)
    }

    /// True for elements whose geometry is a list of points (lines/arrows/freedraw).
    pub fn is_linear_like(&self) -> bool {
        matches!(
            self.kind,
            ElementKind::Line(_) | ElementKind::Arrow(_) | ElementKind::Freedraw(_)
        )
    }

    /// Mark the element mutated. Callers pass the new nonce to keep mutation
    /// side-effect-free / deterministic (no hidden RNG in core).
    pub fn touch(&mut self, version_nonce: u32, updated: Option<u64>) {
        self.version += 1;
        self.version_nonce = version_nonce;
        self.updated = updated;
    }

    /// Convenience: the element type discriminant as a lowercase string,
    /// matching Excalidraw's `type` field for serialization/tooling.
    pub fn type_name(&self) -> &'static str {
        self.kind.type_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_el() -> Element {
        Element::new(
            ElementId::from("a"),
            1,
            10.0,
            20.0,
            100.0,
            40.0,
            ElementKind::Rectangle,
        )
    }

    #[test]
    fn defaults_are_sane() {
        let e = rect_el();
        assert_eq!(e.opacity, 100.0);
        assert_eq!(e.opacity_unit(), 1.0);
        assert!(!e.is_deleted);
        assert_eq!(e.type_name(), "rectangle");
    }

    #[test]
    fn center_and_box() {
        let e = rect_el();
        assert_eq!(e.center(), Point::new(60.0, 40.0));
        assert_eq!(e.raw_box(), Rect::new(10.0, 20.0, 100.0, 40.0));
    }

    #[test]
    fn touch_bumps_version() {
        let mut e = rect_el();
        let v = e.version;
        e.touch(42, Some(1000));
        assert_eq!(e.version, v + 1);
        assert_eq!(e.version_nonce, 42);
        assert_eq!(e.updated, Some(1000));
    }
}
