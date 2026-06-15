//! Bound-text container auto-grow: a container (rectangle/ellipse/diamond) with
//! a bound text label grows so the label always fits.
//!
//! Reimplemented in Rust from Excalidraw's container/bound-text sizing in
//! `packages/excalidraw/element/textElement.ts` — principally
//! `computeContainerDimensionForBoundText`, `getBoundTextElementPosition`,
//! `handleBindTextResize` and `computeBoundTextPosition`. No JavaScript is
//! vendored; the algorithm is ported. See `ATTRIBUTION.md`.
//!
//! These are **pure helpers**: they read a [`Scene`] / [`Element`] and return the
//! change the caller must apply (and record for undo). They never mutate the
//! scene and never touch the editor.
//!
//! ## Sizing model (matching Excalidraw)
//!
//! The label wraps to the container's inner width
//! (`container.width - 2 * BOUND_TEXT_PADDING`). Laying it out yields a
//! [`ContainerTextLayout::required_container_height`], the smallest container
//! height that fits the text without clipping. Excalidraw only ever *grows* a
//! container to its bound text — it never shrinks below the current height — so
//! [`fit_container_to_text`] returns `Some(new_height)` only when the required
//! height exceeds the current height.
//!
//! ## Editor seam
//!
//! After editing a bound text's content, or after resizing a container, the
//! Editor calls [`fit_container_to_text`]; if it returns `Some(h)` the Editor
//! sets the container's `height` to `h`, then calls [`position_bound_text`] to
//! re-place the text element's `(x, y)`, recording both mutations in one undo
//! step. This module performs neither the mutation nor the undo bookkeeping.

use super::{container_text_dimensions, FontSpec, TextAlign, VerticalAlign};
use crate::element::{Element, ElementId, ElementKind};
use crate::geometry::Rect;
use crate::scene::Scene;
use crate::text::TextMeasurer;

/// Build the [`FontSpec`] a text element lays out in, from its [`crate::element::TextData`].
fn text_font(data: &crate::element::TextData) -> FontSpec {
    FontSpec {
        family: data.font_family.clone(),
        size: data.font_size,
        line_height: data.line_height,
    }
}

/// Whether `kind` is a container that can host a bound text label. Mirrors
/// Excalidraw's `isValidContainer` for the shapes this library supports:
/// rectangle, ellipse and diamond.
fn is_text_container(kind: &ElementKind) -> bool {
    matches!(
        kind,
        ElementKind::Rectangle | ElementKind::Ellipse | ElementKind::Diamond
    )
}

/// Find the bound text element for `container` in `scene`.
///
/// Two lookup paths, matching how Excalidraw locates a container's label:
/// 1. the container's `bound_elements` list (an entry whose kind is `Text`), or
/// 2. as a fallback, any live text element whose `container_id` points back at
///    the container.
///
/// The first live text element found is returned. `None` if the container has no
/// bound text.
pub fn bound_text_element<'a>(scene: &'a Scene, container: &Element) -> Option<&'a Element> {
    // Primary: walk the container's declared bound elements.
    for bound in &container.bound_elements {
        if bound.kind == crate::element::BoundElementKind::Text {
            if let Some(el) = scene.get(&bound.id) {
                if !el.is_deleted && matches!(el.kind, ElementKind::Text(_)) {
                    return Some(el);
                }
            }
        }
    }

    // Fallback: scan for a text element pointing back at this container.
    scene.iter_live().find(|el| match &el.kind {
        ElementKind::Text(data) => data.container_id.as_ref() == Some(&container.id),
        _ => false,
    })
}

/// Compute the container height required to fit `text` laid out at
/// `container`'s current width, via `measurer`.
///
/// Returns the [`ContainerTextLayout::required_container_height`]: text block
/// height plus padding top and bottom. This is the raw requirement; it does
/// *not* apply the grow-only rule — see [`fit_container_to_text`].
///
/// The text wraps to the container's inner width, so the alignment used here is
/// the text element's own `text_align` / `vertical_align`.
fn required_height_for_bound_text(
    measurer: &dyn TextMeasurer,
    container: &Element,
    text: &Element,
) -> Option<f64> {
    let data = match &text.kind {
        ElementKind::Text(data) => data,
        _ => return None,
    };

    let font = text_font(data);
    let container_box = container.raw_box();

    let layout = container_text_dimensions(
        measurer,
        container_box,
        &data.text,
        &font,
        data.text_align,
        data.vertical_align,
    );

    Some(layout.required_container_height)
}

/// Decide whether `container_id`'s container should grow to fit its bound text.
///
/// Returns `Some(new_height)` if the container must grow — i.e. the height
/// required to contain the wrapped label (text block + padding top/bottom)
/// exceeds the container's current height. Matches Excalidraw: containers only
/// grow to their text and never shrink below it, so this returns `None` when the
/// text already fits the current height.
///
/// `None` is also returned when:
/// - `container_id` is not in the scene or is not a valid text container,
/// - the container has no (live) bound text element.
///
/// The text is laid out at the container's *current* width via `measurer`, so
/// callers re-running this after a width change get a height that fits the new
/// wrapping.
pub fn fit_container_to_text(
    scene: &Scene,
    container_id: &ElementId,
    measurer: &dyn TextMeasurer,
) -> Option<f64> {
    let container = scene.get(container_id)?;
    if !is_text_container(&container.kind) {
        return None;
    }

    let text = bound_text_element(scene, container)?;
    let required = required_height_for_bound_text(measurer, container, text)?;

    // Grow-only: never shrink below the text. Use a tiny epsilon so identical
    // heights (within float noise) are treated as "already fits".
    if required > container.height + f64::EPSILON {
        Some(required)
    } else {
        None
    }
}

/// The `(x, y)` the bound text element should sit at given the container box and
/// the text's vertical/horizontal alignment, so the label stays inside the
/// container.
///
/// The returned origin is the top-left of the text element's box. Excalidraw
/// keeps bound text inset by [`super::BOUND_TEXT_PADDING`] and centers it by
/// default; `text_align` / `vertical_align` shift the label's box within the
/// padded inner box of the container.
///
/// The text element's own `width`/`height` are used to place it (they are the
/// laid-out block size Excalidraw stores on the text element); they are *not*
/// recomputed here. If the text element is not a text kind, the container's
/// padded top-left is returned.
pub fn position_bound_text(container: &Element, text: &Element) -> (f64, f64) {
    let pad = super::BOUND_TEXT_PADDING;
    let inner_x = container.x + pad;
    let inner_y = container.y + pad;
    let inner_w = (container.width - 2.0 * pad).max(0.0);
    let inner_h = (container.height - 2.0 * pad).max(0.0);

    let (h_align, v_align) = match &text.kind {
        ElementKind::Text(data) => (data.text_align, data.vertical_align),
        _ => (TextAlign::Center, VerticalAlign::Middle),
    };

    let text_w = text.width;
    let text_h = text.height;

    let x_offset = match h_align {
        TextAlign::Left => 0.0,
        TextAlign::Center => (inner_w - text_w) / 2.0,
        TextAlign::Right => inner_w - text_w,
    };
    let y_offset = match v_align {
        VerticalAlign::Top => 0.0,
        VerticalAlign::Middle => (inner_h - text_h) / 2.0,
        VerticalAlign::Bottom => inner_h - text_h,
    };

    (inner_x + x_offset, inner_y + y_offset)
}

/// Convenience used by tests/callers that want the layout-derived text box size
/// for a container's current width (the size Excalidraw would store on the text
/// element after wrapping). Returns `(width, height)` of the wrapped block.
///
/// This is the laid-out block size *without* padding, so callers can set the
/// text element's `width`/`height` before calling [`position_bound_text`].
pub fn bound_text_box_size(
    measurer: &dyn TextMeasurer,
    container: &Element,
    text: &Element,
) -> Option<(f64, f64)> {
    let data = match &text.kind {
        ElementKind::Text(data) => data,
        _ => return None,
    };
    let font = text_font(data);
    let layout = container_text_dimensions(
        measurer,
        container.raw_box(),
        &data.text,
        &font,
        data.text_align,
        data.vertical_align,
    );
    Some((layout.text.width, layout.text.height))
}

/// A container's inner [`Rect`] (the padded region the label lives in). Exposed
/// for callers that need to clip or hit-test against the text area.
pub fn container_inner_rect(container: &Element) -> Rect {
    let pad = super::BOUND_TEXT_PADDING;
    Rect::new(
        container.x + pad,
        container.y + pad,
        (container.width - 2.0 * pad).max(0.0),
        (container.height - 2.0 * pad).max(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{BoundElement, TextData};
    use crate::geometry::Point;
    use crate::text::{FontFamily, MonospaceMeasurer};

    /// A measurer whose advance is exactly 1 unit per char per font-size unit,
    /// so assertion arithmetic is exact.
    fn unit_measurer() -> MonospaceMeasurer {
        MonospaceMeasurer { advance_ratio: 1.0 }
    }

    fn container(id: &str, w: f64, h: f64) -> Element {
        Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            w,
            h,
            ElementKind::Rectangle,
        )
    }

    /// A bound text element with `unit_measurer`-friendly font (size 10 => char
    /// width 10), the given text, centered alignment, bound to `container_id`.
    fn bound_text(id: &str, container_id: &str, text: &str) -> Element {
        let mut data = TextData::new(text);
        data.font_family = FontFamily::Code;
        data.font_size = 10.0;
        data.line_height = 1.25; // line_spacing = 12.5
        data.text_align = TextAlign::Center;
        data.vertical_align = VerticalAlign::Middle;
        data.container_id = Some(ElementId::from(container_id));
        Element::new(
            ElementId::from(id),
            2,
            0.0,
            0.0,
            0.0,
            0.0,
            ElementKind::Text(data),
        )
    }

    /// Wire a container <-> text pair into a scene, with the text's box sized to
    /// the wrapped block so positioning is meaningful.
    fn scene_with_pair(c: &mut Element, t: &mut Element, m: &dyn TextMeasurer) -> Scene {
        c.bound_elements.push(BoundElement {
            id: t.id.clone(),
            kind: crate::element::BoundElementKind::Text,
        });
        if let Some((w, h)) = bound_text_box_size(m, c, t) {
            t.width = w;
            t.height = h;
        }
        let mut s = Scene::new();
        s.insert(c.clone());
        s.insert(t.clone());
        s
    }

    #[test]
    fn one_line_label_needs_no_grow() {
        let m = unit_measurer();
        // Container 200 wide, tall enough: one line "ab" -> block 12.5,
        // required = 12.5 + 10 = 22.5. Height 100 already fits.
        let mut c = container("c", 200.0, 100.0);
        let mut t = bound_text("t", "c", "ab");
        let s = scene_with_pair(&mut c, &mut t, &m);
        assert_eq!(fit_container_to_text(&s, &ElementId::from("c"), &m), None);
    }

    #[test]
    fn long_multiline_label_grows_container() {
        let m = unit_measurer();
        // Container 200 wide, only 20 tall. Text has 3 explicit lines ->
        // block 3 * 12.5 = 37.5, required = 37.5 + 10 = 47.5 > 20 -> grow.
        let mut c = container("c", 200.0, 20.0);
        let mut t = bound_text("t", "c", "aa\nbb\ncc");
        let s = scene_with_pair(&mut c, &mut t, &m);
        let grown = fit_container_to_text(&s, &ElementId::from("c"), &m);
        assert_eq!(grown, Some(47.5));
    }

    #[test]
    fn wrapping_long_line_grows_container() {
        let m = unit_measurer();
        // Container width 70 -> inner width 60. "aaa bbb" (70) wraps to 2 lines.
        // block = 2 * 12.5 = 25, required = 25 + 10 = 35. Container only 15 tall.
        let mut c = container("c", 70.0, 15.0);
        let mut t = bound_text("t", "c", "aaa bbb");
        let s = scene_with_pair(&mut c, &mut t, &m);
        let grown = fit_container_to_text(&s, &ElementId::from("c"), &m);
        assert_eq!(grown, Some(35.0));
    }

    #[test]
    fn grows_only_never_shrinks() {
        let m = unit_measurer();
        // Tall container, short text: required height is far below current.
        let mut c = container("c", 200.0, 500.0);
        let mut t = bound_text("t", "c", "ab");
        let s = scene_with_pair(&mut c, &mut t, &m);
        // Never returns a shrink.
        assert_eq!(fit_container_to_text(&s, &ElementId::from("c"), &m), None);
    }

    #[test]
    fn finds_bound_text_via_container_id_fallback() {
        let m = unit_measurer();
        // No bound_elements entry; only the text's container_id links back.
        let c = container("c", 200.0, 10.0);
        let mut t = bound_text("t", "c", "aa\nbb\ncc");
        // Size the text box but DON'T add to bound_elements.
        if let Some((w, h)) = bound_text_box_size(&m, &c, &t) {
            t.width = w;
            t.height = h;
        }
        let mut s = Scene::new();
        s.insert(c.clone());
        s.insert(t.clone());
        // Fallback scan should still find it and grow.
        assert!(fit_container_to_text(&s, &ElementId::from("c"), &m).is_some());
        assert!(bound_text_element(&s, &c).is_some());
    }

    #[test]
    fn non_container_returns_none() {
        let m = unit_measurer();
        // An arrow is not a valid text container.
        let arrow = Element::new(
            ElementId::from("a"),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Arrow(crate::element::LinearData::arrow(vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
            ])),
        );
        let mut s = Scene::new();
        s.insert(arrow);
        assert_eq!(fit_container_to_text(&s, &ElementId::from("a"), &m), None);
    }

    #[test]
    fn container_without_bound_text_returns_none() {
        let m = unit_measurer();
        let c = container("c", 100.0, 50.0);
        let mut s = Scene::new();
        s.insert(c);
        assert_eq!(fit_container_to_text(&s, &ElementId::from("c"), &m), None);
    }

    #[test]
    fn positioned_text_origin_stays_within_container() {
        let m = unit_measurer();
        let mut c = container("c", 200.0, 120.0);
        let mut t = bound_text("t", "c", "aa\nbb");
        let _ = scene_with_pair(&mut c, &mut t, &m);

        let (x, y) = position_bound_text(&c, &t);

        // Origin must be inside the padded inner box.
        let inner = container_inner_rect(&c);
        assert!(x >= inner.x - 1e-9, "x {x} < inner.x {}", inner.x);
        assert!(y >= inner.y - 1e-9, "y {y} < inner.y {}", inner.y);
        // And the text box must not overflow the inner box.
        assert!(
            x + t.width <= inner.x + inner.width + 1e-9,
            "text right edge overflows"
        );
        assert!(
            y + t.height <= inner.y + inner.height + 1e-9,
            "text bottom edge overflows"
        );
    }

    #[test]
    fn position_centers_by_default() {
        let m = unit_measurer();
        // Container 200x120. Text "aa\nbb": width 20, height 25.
        let mut c = container("c", 200.0, 120.0);
        let mut t = bound_text("t", "c", "aa\nbb");
        let _ = scene_with_pair(&mut c, &mut t, &m);
        assert_eq!((t.width, t.height), (20.0, 25.0));

        let (x, y) = position_bound_text(&c, &t);
        // inner box: x=5,y=5,w=190,h=110.
        // x_offset = (190 - 20)/2 = 85 -> x = 5 + 85 = 90.
        // y_offset = (110 - 25)/2 = 42.5 -> y = 5 + 42.5 = 47.5.
        assert_eq!(x, 90.0);
        assert_eq!(y, 47.5);
    }

    #[test]
    fn position_top_left_aligned() {
        let m = unit_measurer();
        let mut c = container("c", 200.0, 120.0);
        let mut t = bound_text("t", "c", "aa");
        if let ElementKind::Text(data) = &mut t.kind {
            data.text_align = TextAlign::Left;
            data.vertical_align = VerticalAlign::Top;
        }
        let _ = scene_with_pair(&mut c, &mut t, &m);
        let (x, y) = position_bound_text(&c, &t);
        // Top-left: just the padding.
        assert_eq!(x, super::super::BOUND_TEXT_PADDING + 0.0);
        assert_eq!(y, super::super::BOUND_TEXT_PADDING + 0.0);
    }

    #[test]
    fn grow_then_fits_after_applying() {
        let m = unit_measurer();
        // Simulate the editor seam: grow, apply, re-check returns None.
        let mut c = container("c", 200.0, 10.0);
        let mut t = bound_text("t", "c", "aa\nbb\ncc");
        let mut s = scene_with_pair(&mut c, &mut t, &m);
        let cid = ElementId::from("c");

        let new_h = fit_container_to_text(&s, &cid, &m).expect("should grow");
        // Apply.
        s.get_mut(&cid).unwrap().height = new_h;
        // Re-check: now it fits exactly, so no further grow.
        assert_eq!(fit_container_to_text(&s, &cid, &m), None);
    }
}
