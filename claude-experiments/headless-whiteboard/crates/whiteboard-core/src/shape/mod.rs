//! Element → geometry path generation.
//!
//! Turns an [`Element`] into the vector [`Path`]s that represent its outline and
//! (for fillable shapes) its fill. Two modes:
//!
//! - **Clean** — precise geometry (exact rects, Bézier ellipses, straight lines).
//! - **Rough** — hand-drawn geometry via the [`crate::rough`] generator, driven
//!   by the element's `seed`/`roughness`.
//!
//! Phase 1 implements the per-element generators and the rough delegation. This
//! file defines the [`ShapeGeometry`]/[`ShapeGenerator`] vocabulary the
//! tessellator (render module) calls; the per-element clean geometry lives in
//! [`clean`], arrowheads in [`arrowhead`], and the (delegating-for-now)
//! hand-drawn generator in [`rough_gen`].

mod arrowhead;
mod clean;
mod elbow;
mod rough_gen;

pub use arrowhead::{arrowhead_geometry, arrowhead_paths, ArrowheadGeometry};
pub use clean::{
    catmull_rom_path, clean_geometry, diamond_path, ellipse_path, rounded_rectangle_path,
    roundness_radius,
};
pub use elbow::{elbow_geometry, elbow_route};
pub use rough_gen::RoughGenerator;

use crate::element::{Element, Roundness};
use crate::geometry::Path;

/// The outline + optional fill geometry for one element, in the element's own
/// unrotated coordinate space (the tessellator applies rotation/translation).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapeGeometry {
    /// Closed/open path(s) to stroke as the element's outline.
    pub outline: Vec<Path>,
    /// Closed region(s) to flood-fill with the background color (solid fill).
    pub fill: Vec<Path>,
    /// Open path(s) to *stroke* with the background color — the hachure /
    /// cross-hatch / zigzag fill lines. Stroked rather than filled because each
    /// is a line, not a region. Empty for solid fills.
    pub fill_strokes: Vec<Path>,
    /// Closed region(s) to flood-fill with the element's **stroke** color (not
    /// the background). Used for solid arrowheads (filled triangle / dot /
    /// diamond), which are colored by the line's stroke.
    pub fill_with_stroke: Vec<Path>,
}

impl ShapeGeometry {
    pub fn outline_only(path: Path) -> Self {
        ShapeGeometry {
            outline: vec![path],
            fill: Vec::new(),
            fill_strokes: Vec::new(),
            fill_with_stroke: Vec::new(),
        }
    }
}

/// Produces geometry for elements. Implemented by `Clean` and (Phase 1) `Rough`.
pub trait ShapeGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry;
}

/// Precise, non-sketchy geometry generator.
///
/// Supports every element type via [`clean::clean_geometry`] with sharp corners.
/// For rounded box/diamond corners use [`RoundedCleanGenerator`] (the shared
/// `Element` type carries no roundness field, so the radius is configured on the
/// generator rather than read off the element).
#[derive(Debug, Clone, Copy, Default)]
pub struct CleanGenerator;

impl ShapeGenerator for CleanGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry {
        clean_geometry(element, None)
    }
}

/// Clean geometry generator that rounds box/diamond corners with a configured
/// [`Roundness`]. Kept separate from [`CleanGenerator`] so the latter stays a
/// zero-field unit type usable as a bare value.
#[derive(Debug, Clone, Copy)]
pub struct RoundedCleanGenerator {
    pub roundness: Roundness,
}

impl RoundedCleanGenerator {
    pub fn new(roundness: Roundness) -> Self {
        RoundedCleanGenerator { roundness }
    }
}

impl ShapeGenerator for RoundedCleanGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry {
        clean_geometry(element, Some(self.roundness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, ElementKind, RoundnessKind};

    #[test]
    fn clean_rectangle_outline() {
        let e = Element::new(
            ElementId::from("r"),
            1,
            0.0,
            0.0,
            20.0,
            10.0,
            ElementKind::Rectangle,
        );
        let g = CleanGenerator.geometry(&e);
        assert_eq!(g.outline.len(), 1);
        // 4 corners (move + 3 line) + close = 5 segments.
        assert_eq!(g.outline[0].segments.len(), 5);
    }

    #[test]
    fn transparent_rectangle_has_no_fill() {
        let e = Element::new(
            ElementId::from("r"),
            1,
            0.0,
            0.0,
            20.0,
            10.0,
            ElementKind::Rectangle,
        );
        assert!(CleanGenerator.geometry(&e).fill.is_empty());
    }

    #[test]
    fn rounded_generator_rounds_corners() {
        let e = Element::new(
            ElementId::from("r"),
            1,
            0.0,
            0.0,
            80.0,
            40.0,
            ElementKind::Rectangle,
        );
        let rnd = Roundness {
            kind: RoundnessKind::AdaptiveRadius,
            value: Some(8.0),
        };
        let g = RoundedCleanGenerator::new(rnd).geometry(&e);
        assert!(g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, crate::geometry::PathSegment::CubicTo { .. })));
    }
}
