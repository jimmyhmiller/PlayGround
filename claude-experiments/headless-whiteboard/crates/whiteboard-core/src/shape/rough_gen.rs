//! Hand-drawn ("rough") per-element geometry generator.
//!
//! Reimplemented from Excalidraw's `generateRoughOptions` + `ShapeCache`
//! (`packages/element/src/shape.ts`), which feed each element into Rough.js.
//!
//! Dispatches each element through the Rough.js *drawable* generators in
//! [`crate::rough`] (jittered rectangle / ellipse / polygon / linear path) for
//! the outline, and [`crate::rough::fill_polygon`] for the patterned fill. The
//! result is the sketchy Excalidraw look, deterministic per the element's
//! `seed`. Freedraw is already hand-drawn by nature and arrowheads stay crisp,
//! so those reuse the clean geometry. When `roughness == 0` the whole element
//! falls back to clean geometry for precise output.

use crate::element::{Element, ElementKind, Roundness};
use crate::geometry::Point;
use crate::render::FillStyle;
use crate::rough::{
    fill_polygon, rough_ellipse, rough_linear_path, rough_polygon, rough_rectangle, RoughOptions,
};

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
        // A roughness of zero means "draw it precisely" — no point jittering.
        if element.roughness <= 0.0 {
            return clean_geometry(element, self.roundness);
        }
        let opts = self.options(element);
        let w = element.width;
        let h = element.height;

        // Start from the clean geometry: it already handles every element type
        // correctly, gives us arrowheads, and is the fallback for kinds whose
        // outline is not a simple jitterable primitive (freedraw, text, the
        // fill region we re-pattern below).
        let mut clean = clean_geometry(element, self.roundness);

        // Replace the outline with a sketchy one where a rough primitive applies.
        // Freedraw / text keep their clean outline (freedraw is already
        // hand-drawn; text has no outline). For lines/arrows we roughen the body
        // and KEEP the clean arrowhead paths (everything after the first outline
        // path produced by the clean linear generator is an arrowhead).
        match &element.kind {
            ElementKind::Rectangle
            | ElementKind::Image(_)
            | ElementKind::Frame(_)
            | ElementKind::Selection
                if self.roundness.is_none() =>
            {
                clean.outline = vec![rough_rectangle(w, h, &opts)];
            }
            ElementKind::Ellipse => {
                clean.outline = vec![rough_ellipse(w, h, &opts)];
            }
            ElementKind::Diamond if self.roundness.is_none() => {
                clean.outline = vec![rough_polygon(&diamond_points(w, h), &opts)];
            }
            ElementKind::Line(data) | ElementKind::Arrow(data) => {
                if data.points.len() >= 2 {
                    let body = if data.polygon {
                        rough_polygon(&data.points, &opts)
                    } else {
                        rough_linear_path(&data.points, &opts)
                    };
                    // outline[0] is the clean body; keep the rest (arrowheads).
                    if clean.outline.is_empty() {
                        clean.outline = vec![body];
                    } else {
                        clean.outline[0] = body;
                    }
                }
            }
            // Rounded box/diamond, freedraw, text: keep the clean outline.
            _ => {}
        }

        // Re-pattern the fill as a sketchy hachure/cross-hatch/zigzag/dots fill
        // when the element is fillable with a non-solid style. These are stroked
        // lines, not a region, so they go in `fill_strokes` (the tessellator
        // strokes them with the background color). Solid fills keep the clean
        // flood-fill region in `fill`.
        if !clean.fill.is_empty() && element.fill_style != FillStyle::Solid {
            if let Some(region) = fillable_polygon(element) {
                let lines = fill_polygon(&region, element.fill_style, &opts);
                if !lines.is_empty() {
                    clean.fill.clear();
                    clean.fill_strokes = lines;
                }
            }
        }

        clean
    }
}

/// The four edge-midpoint vertices of a diamond inscribed in the `w`×`h` box,
/// in element-local space. Matches the clean diamond generator's vertices.
fn diamond_points(w: f64, h: f64) -> Vec<Point> {
    vec![
        Point::new(w / 2.0, 0.0),
        Point::new(w, h / 2.0),
        Point::new(w / 2.0, h),
        Point::new(0.0, h / 2.0),
    ]
}

/// The polygon (element-local) whose interior the fill pattern should cover, for
/// fillable element kinds. `None` for kinds that are not pattern-fillable.
fn fillable_polygon(element: &Element) -> Option<Vec<Point>> {
    let w = element.width;
    let h = element.height;
    match &element.kind {
        ElementKind::Rectangle | ElementKind::Image(_) | ElementKind::Frame(_) => Some(vec![
            Point::new(0.0, 0.0),
            Point::new(w, 0.0),
            Point::new(w, h),
            Point::new(0.0, h),
        ]),
        ElementKind::Diamond => Some(diamond_points(w, h)),
        // Ellipse fill is approximated by its inscribed polygon perimeter; the
        // fill module clips hachure lines to it.
        ElementKind::Ellipse => Some(ellipse_polygon(w, h, 48)),
        ElementKind::Line(data) if data.polygon => Some(data.points.clone()),
        _ => None,
    }
}

/// A closed polygon approximating the ellipse perimeter (element-local).
fn ellipse_polygon(w: f64, h: f64, samples: usize) -> Vec<Point> {
    let (cx, cy) = (w / 2.0, h / 2.0);
    let (rx, ry) = (w / 2.0, h / 2.0);
    (0..samples)
        .map(|i| {
            let t = (i as f64) / (samples as f64) * std::f64::consts::TAU;
            Point::new(cx + rx * t.cos(), cy + ry * t.sin())
        })
        .collect()
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
    fn zero_roughness_matches_clean() {
        let e = rect(7, 0.0);
        let rough = RoughGenerator::new().geometry(&e);
        let clean = clean_geometry(&e, None);
        assert_eq!(rough, clean, "roughness 0 must be precise");
    }

    #[test]
    fn nonzero_roughness_diverges_from_clean() {
        let e = rect(7, 1.0);
        let rough = RoughGenerator::new().geometry(&e);
        let clean = clean_geometry(&e, None);
        assert_ne!(
            rough.outline, clean.outline,
            "roughness 1 must jitter the outline"
        );
    }

    #[test]
    fn rough_is_seed_deterministic() {
        let e = rect(7, 1.5);
        let a = RoughGenerator::new().geometry(&e);
        let b = RoughGenerator::new().geometry(&e);
        assert_eq!(a, b, "same seed => identical sketch");
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
