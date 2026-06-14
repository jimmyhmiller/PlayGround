//! Hand-drawn ("rough") per-element geometry generator.
//!
//! Reimplemented from Excalidraw's `generateRoughOptions` + `ShapeCache`
//! (`packages/element/src/shape.ts`), which feed each element into Rough.js.
//!
//! The Rough.js *drawable* generators (jittered line / ellipse / polygon / curve
//! plus the hachure/cross-hatch/zigzag/dots fills) are being built in parallel
//! under [`crate::rough`]. Until those land, this generator produces the exact
//! same correct outline/fill geometry as [`super::clean`] so the render pipeline
//! is fully functional for every element type — never a fake or empty shape for
//! a real element. The seam to swap in the sketchy geometry is marked with
//! `TODO(rough)` at each call site.

use crate::element::{Element, Roundness};
use crate::rough::RoughOptions;

use super::clean::clean_geometry;
use super::{ShapeGenerator, ShapeGeometry};

/// Generator that produces hand-drawn geometry driven by an element's
/// `seed`/`roughness`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoughGenerator {
    /// Optional corner roundness applied to box/diamond outlines. The shared
    /// `Element` type carries no roundness field yet, so callers configure it
    /// here (mirrors [`super::clean::clean_geometry`]'s `roundness` argument).
    pub roundness: Option<Roundness>,
}

impl RoughGenerator {
    pub fn new() -> Self {
        RoughGenerator::default()
    }

    pub fn with_roundness(roundness: Roundness) -> Self {
        RoughGenerator {
            roundness: Some(roundness),
        }
    }

    /// Build the Rough.js options for `element` (seed + roughness), used once the
    /// sketchy generators are wired in.
    fn options(&self, element: &Element) -> RoughOptions {
        RoughOptions::for_element(element.roughness, element.seed)
    }
}

impl ShapeGenerator for RoughGenerator {
    fn geometry(&self, element: &Element) -> ShapeGeometry {
        // Compute the deterministic options now so the seam is obvious and the
        // RNG plumbing is exercised even before the sketchy paths exist.
        let _opts = self.options(element);

        // TODO(rough): once `crate::rough` exposes its drawable generators,
        // dispatch on `element.kind` here and build sketchy geometry:
        //   - Rectangle/Image/Frame  -> rough::rectangle(w, h, &_opts)
        //   - Ellipse                -> rough::ellipse(w, h, &_opts)
        //   - Diamond                -> rough::polygon(&diamond_pts, &_opts)
        //   - Line/Arrow             -> rough::linear_path(points, &_opts) (+ clean arrowheads)
        //   - Freedraw               -> perfect-freehand outline (pressure aware)
        //   - fills                  -> rough::fill(path, element.fill_style, &_opts)
        // Until then we return the exact clean geometry — correct, never fake —
        // so every element type renders.
        clean_geometry(element, self.roundness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, ElementKind, LinearData, RoundnessKind};
    use crate::geometry::Point;

    fn rect(seed: u32, roughness: f64) -> Element {
        let mut e = Element::new(
            ElementId::from("r"),
            seed,
            0.0,
            0.0,
            20.0,
            10.0,
            ElementKind::Rectangle,
        );
        e.roughness = roughness;
        e
    }

    #[test]
    fn rough_matches_clean_for_now() {
        let e = rect(7, 1.0);
        let rough = RoughGenerator::new().geometry(&e);
        let clean = clean_geometry(&e, None);
        assert_eq!(rough, clean);
    }

    #[test]
    fn rough_never_empty_for_real_element() {
        // An arrow must always yield geometry (body + head), never a fake stub.
        let data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(20.0, 0.0)]);
        let e = Element::new(
            ElementId::from("a"),
            3,
            0.0,
            0.0,
            20.0,
            0.0,
            ElementKind::Arrow(data),
        );
        let g = RoughGenerator::new().geometry(&e);
        assert!(!g.outline.is_empty());
    }

    #[test]
    fn rough_honors_roundness_seam() {
        let rnd = Roundness {
            kind: RoundnessKind::AdaptiveRadius,
            value: Some(6.0),
        };
        let g = RoughGenerator::with_roundness(rnd).geometry(&rect(1, 1.0));
        // Rounded ⇒ corner cubics present.
        assert!(g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, crate::geometry::PathSegment::CubicTo { .. })));
    }

    #[test]
    fn options_are_seed_deterministic() {
        let e = rect(123, 2.0);
        let g = RoughGenerator::new();
        let a = g.options(&e);
        let b = g.options(&e);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.seed, 123);
        assert_eq!(a.roughness, 2.0);
    }
}
