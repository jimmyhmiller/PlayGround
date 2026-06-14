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
//! file defines the [`ShapeGenerator`] trait that the tessellator (render module)
//! calls, plus the clean rectangle generator as a worked, tested baseline so the
//! pipeline is exercisable end to end before the rest lands.

use crate::element::{Element, ElementKind};
use crate::geometry::{Path, Point};

/// The outline + optional fill geometry for one element, in the element's own
/// unrotated coordinate space (the tessellator applies rotation/translation).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapeGeometry {
    /// Closed/open path(s) to stroke as the element's outline.
    pub outline: Vec<Path>,
    /// Path(s) describing the fill region (empty if not filled).
    pub fill: Vec<Path>,
}

impl ShapeGeometry {
    pub fn outline_only(path: Path) -> Self {
        ShapeGeometry {
            outline: vec![path],
            fill: Vec::new(),
        }
    }
}

/// Produces geometry for elements. Implemented by `Clean` and (Phase 1) `Rough`.
pub trait ShapeGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry;
}

/// Precise, non-sketchy geometry generator.
#[derive(Debug, Clone, Copy, Default)]
pub struct CleanGenerator;

impl ShapeGenerator for CleanGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry {
        match &element.kind {
            ElementKind::Rectangle | ElementKind::Frame(_) | ElementKind::Selection => {
                let w = element.width;
                let h = element.height;
                let path = Path::polygon(&[
                    Point::new(0.0, 0.0),
                    Point::new(w, 0.0),
                    Point::new(w, h),
                    Point::new(0.0, h),
                ]);
                let mut g = ShapeGeometry::outline_only(path.clone());
                if element.kind.is_fillable() && !element.background_color.is_transparent() {
                    g.fill = vec![path];
                }
                g
            }
            // Remaining element types are generated in Phase 1. Returning empty
            // geometry here is honest (nothing to draw yet) rather than a fake
            // placeholder shape; the tessellator simply emits no commands for it.
            _ => ShapeGeometry::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementId;

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
}
