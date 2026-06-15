//! Property mutation as a pure scene operation.
//!
//! Applies a single style change across a set of elements, mirroring the
//! per-property updates Excalidraw performs from its property panel
//! (`packages/excalidraw/actions/actionProperties.tsx`). This is reimplemented
//! in Rust — no JavaScript is vendored. The operation is intentionally minimal:
//! it sets the matching field on each live, present element and reports whether
//! anything actually changed. It does not bump element versions or seeds; the
//! editor layer owns undo snapshotting and version bookkeeping.

use crate::element::ElementId;
use crate::render::{Color, FillStyle, StrokeStyle};

use super::Scene;

/// A single style property change to apply to a set of elements.
///
/// Each variant maps to exactly one field on [`crate::element::Element`].
/// Numeric variants are clamped to their valid ranges by [`apply_style`]:
/// `Opacity` to `0..=100` (Excalidraw stores opacity as a percentage) and
/// `StrokeWidth` to `>= 0`.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleChange {
    StrokeColor(Color),
    BackgroundColor(Color),
    FillStyle(FillStyle),
    StrokeWidth(f64),
    StrokeStyle(StrokeStyle),
    Roughness(f64),
    Opacity(f64),
}

/// Apply `change` to every element in `ids` that exists and is not deleted.
///
/// Missing ids and soft-deleted elements are skipped. Returns whether any field
/// was actually modified (i.e. at least one element existed, was live, and its
/// value differed from the requested value).
///
/// Numeric values are normalized before comparison/assignment:
/// - `Opacity` is clamped to `0.0..=100.0`.
/// - `StrokeWidth` is clamped to `>= 0.0` (no negative widths).
/// - `Roughness` is taken as-is (Excalidraw uses 0/1/2 but allows any f64).
pub fn apply_style(scene: &mut Scene, ids: &[ElementId], change: &StyleChange) -> bool {
    let mut changed = false;
    for id in ids {
        let Some(el) = scene.get_mut(id) else {
            continue;
        };
        if el.is_deleted {
            continue;
        }
        match change {
            StyleChange::StrokeColor(c) => {
                if el.stroke_color != *c {
                    el.stroke_color = *c;
                    changed = true;
                }
            }
            StyleChange::BackgroundColor(c) => {
                if el.background_color != *c {
                    el.background_color = *c;
                    changed = true;
                }
            }
            StyleChange::FillStyle(f) => {
                if el.fill_style != *f {
                    el.fill_style = *f;
                    changed = true;
                }
            }
            StyleChange::StrokeWidth(w) => {
                let w = w.max(0.0);
                if el.stroke_width != w {
                    el.stroke_width = w;
                    changed = true;
                }
            }
            StyleChange::StrokeStyle(s) => {
                if el.stroke_style != *s {
                    el.stroke_style = *s;
                    changed = true;
                }
            }
            StyleChange::Roughness(r) => {
                if el.roughness != *r {
                    el.roughness = *r;
                    changed = true;
                }
            }
            StyleChange::Opacity(o) => {
                let o = o.clamp(0.0, 100.0);
                if el.opacity != o {
                    el.opacity = o;
                    changed = true;
                }
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    fn el(id: &str) -> Element {
        Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Rectangle,
        )
    }

    fn scene_with(ids: &[&str]) -> Scene {
        let mut s = Scene::new();
        for id in ids {
            s.insert(el(id));
        }
        s
    }

    fn id(s: &str) -> ElementId {
        ElementId::from(s)
    }

    #[test]
    fn stroke_color_applies_to_all() {
        let mut s = scene_with(&["a", "b"]);
        let red = Color::rgb(255, 0, 0);
        let changed = apply_style(&mut s, &[id("a"), id("b")], &StyleChange::StrokeColor(red));
        assert!(changed);
        assert_eq!(s.get(&id("a")).unwrap().stroke_color, red);
        assert_eq!(s.get(&id("b")).unwrap().stroke_color, red);
    }

    #[test]
    fn background_color_applies() {
        let mut s = scene_with(&["a"]);
        let blue = Color::rgb(0, 0, 255);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::BackgroundColor(blue)
        ));
        assert_eq!(s.get(&id("a")).unwrap().background_color, blue);
    }

    #[test]
    fn fill_style_applies() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::FillStyle(FillStyle::Solid)
        ));
        assert_eq!(s.get(&id("a")).unwrap().fill_style, FillStyle::Solid);
    }

    #[test]
    fn stroke_style_applies() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::StrokeStyle(StrokeStyle::Dashed)
        ));
        assert_eq!(s.get(&id("a")).unwrap().stroke_style, StrokeStyle::Dashed);
    }

    #[test]
    fn roughness_applies() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::Roughness(2.0)
        ));
        assert_eq!(s.get(&id("a")).unwrap().roughness, 2.0);
    }

    #[test]
    fn opacity_is_clamped_high() {
        let mut s = scene_with(&["a"]);
        // Move off the default (100.0) first so the clamp-to-100 is observable.
        apply_style(&mut s, &[id("a")], &StyleChange::Opacity(40.0));
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::Opacity(250.0)
        ));
        assert_eq!(s.get(&id("a")).unwrap().opacity, 100.0);
    }

    #[test]
    fn opacity_is_clamped_low() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::Opacity(-30.0)
        ));
        assert_eq!(s.get(&id("a")).unwrap().opacity, 0.0);
    }

    #[test]
    fn stroke_width_is_clamped_nonnegative() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::StrokeWidth(-5.0)
        ));
        assert_eq!(s.get(&id("a")).unwrap().stroke_width, 0.0);
    }

    #[test]
    fn stroke_width_sets_positive() {
        let mut s = scene_with(&["a"]);
        assert!(apply_style(
            &mut s,
            &[id("a")],
            &StyleChange::StrokeWidth(4.0)
        ));
        assert_eq!(s.get(&id("a")).unwrap().stroke_width, 4.0);
    }

    #[test]
    fn no_change_when_value_matches() {
        let mut s = scene_with(&["a"]);
        // Default opacity is already 100.0.
        let changed = apply_style(&mut s, &[id("a")], &StyleChange::Opacity(100.0));
        assert!(!changed);
    }

    #[test]
    fn missing_ids_are_skipped() {
        let mut s = scene_with(&["a"]);
        let red = Color::rgb(255, 0, 0);
        // Only "missing" requested; nothing exists to change.
        let changed = apply_style(&mut s, &[id("missing")], &StyleChange::StrokeColor(red));
        assert!(!changed);
    }

    #[test]
    fn deleted_elements_are_skipped() {
        let mut s = Scene::new();
        let mut d = el("a");
        d.is_deleted = true;
        s.insert(d);
        let red = Color::rgb(255, 0, 0);
        let changed = apply_style(&mut s, &[id("a")], &StyleChange::StrokeColor(red));
        assert!(!changed);
        // The deleted element is untouched.
        assert_ne!(s.get(&id("a")).unwrap().stroke_color, red);
    }

    #[test]
    fn partial_change_still_reports_true() {
        // "a" already at target, "b" differs -> overall change is true.
        let mut s = scene_with(&["a", "b"]);
        let red = Color::rgb(255, 0, 0);
        apply_style(&mut s, &[id("a")], &StyleChange::StrokeColor(red));
        let changed = apply_style(&mut s, &[id("a"), id("b")], &StyleChange::StrokeColor(red));
        assert!(changed); // because of "b"
    }
}
