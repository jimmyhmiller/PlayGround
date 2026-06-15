//! The selection UI, emitted as backend-neutral [`DrawCommand`]s.
//!
//! Reimplemented from Excalidraw's interactive-canvas selection rendering
//! (`packages/excalidraw/renderer/interactiveScene.ts`:
//! `renderSelectionElement` / `renderTransformHandles`, and the marquee in
//! `renderSelectionBorder`). Excalidraw draws the selection bounding box, the
//! eight square resize handles, the circular rotation handle with its connector
//! stub, and the drag-marquee rectangle directly on a separate canvas overlaid
//! on the scene. We produce the equivalent geometry as a [`RenderScene`].
//!
//! Everything here is in **screen** space: the [`HandleLayout`] already maps the
//! selection box through the viewport and applies a fixed pixel handle size and
//! rotation offset, so the commands this module emits must be appended *after*
//! the (viewport-transformed) scene, with no further transform.

use super::{Color, DrawCommand, Paint, RenderScene, Stroke, StrokeStyle};
use crate::geometry::{Path, Point, Rect};
use crate::interaction::{Handle, HandleLayout, HANDLE_SIZE};

/// Colors and sizes for the selection overlay. Defaults mirror Excalidraw's
/// light-theme selection accent (`#6965db`/`oc-blue` family — we use the
/// canonical `#4263eb` selection blue) with white handle fills.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayStyle {
    /// Accent color of the bounding box, handle borders and rotation handle.
    pub accent: Color,
    /// Fill color of the square resize handles and the rotation circle.
    pub handle_fill: Color,
    /// Side length (screen px) of each square resize handle. Defaults to
    /// [`HANDLE_SIZE`].
    pub handle_size: f64,
    /// Radius (screen px) of the round rotation handle.
    pub rotation_radius: f64,
    /// Stroke width (screen px) of the bounding box border.
    pub bbox_width: f64,
    /// Stroke width (screen px) of handle borders and the rotation connector.
    pub handle_stroke_width: f64,
    /// Border color of the marquee rect.
    pub marquee_stroke: Color,
    /// Faint interior fill of the marquee rect.
    pub marquee_fill: Color,
    /// Stroke width (screen px) of the marquee border.
    pub marquee_width: f64,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        // Excalidraw selection blue.
        let accent = Color::rgb(0x42, 0x63, 0xeb);
        OverlayStyle {
            accent,
            handle_fill: Color::WHITE,
            handle_size: HANDLE_SIZE,
            rotation_radius: HANDLE_SIZE / 2.0,
            bbox_width: 1.0,
            handle_stroke_width: 1.0,
            marquee_stroke: accent,
            // ~7% alpha wash, like Excalidraw's translucent marquee.
            marquee_fill: Color::rgba(0x42, 0x63, 0xeb, 0x14),
            marquee_width: 1.0,
        }
    }
}

/// Magic constant for approximating a quarter circle with a cubic Bézier:
/// `4/3 * tan(pi/8)`. Each of the four arcs uses control points offset from the
/// endpoints by `radius * KAPPA` along the tangent.
const KAPPA: f64 = 0.552_284_749_830_793_4;

/// Build a closed [`Path`] approximating a circle of `radius` centered at `c`
/// from four cubic-Bézier quarter arcs (there is no circle primitive).
fn circle_path(c: Point, radius: f64) -> Path {
    let r = radius;
    let k = radius * KAPPA;
    let mut b = Path::builder();
    // Start at the rightmost point and sweep clockwise.
    let right = Point::new(c.x + r, c.y);
    let bottom = Point::new(c.x, c.y + r);
    let left = Point::new(c.x - r, c.y);
    let top = Point::new(c.x, c.y - r);
    b.move_to(right);
    // right -> bottom
    b.cubic_to(
        Point::new(c.x + r, c.y + k),
        Point::new(c.x + k, c.y + r),
        bottom,
    );
    // bottom -> left
    b.cubic_to(
        Point::new(c.x - k, c.y + r),
        Point::new(c.x - r, c.y + k),
        left,
    );
    // left -> top
    b.cubic_to(
        Point::new(c.x - r, c.y - k),
        Point::new(c.x - k, c.y - r),
        top,
    );
    // top -> right
    b.cubic_to(
        Point::new(c.x + k, c.y - r),
        Point::new(c.x + r, c.y - k),
        right,
    );
    b.close();
    b.build()
}

/// A square handle centered at `c` with the given side length, as a closed path.
fn square_path(c: Point, side: f64) -> Path {
    let h = side / 2.0;
    Path::polygon(&[
        Point::new(c.x - h, c.y - h),
        Point::new(c.x + h, c.y - h),
        Point::new(c.x + h, c.y + h),
        Point::new(c.x - h, c.y + h),
    ])
}

/// Closed path of an axis-aligned rect (used for the marquee).
fn rect_path(r: Rect) -> Path {
    Path::polygon(&[
        r.top_left(),
        r.top_right(),
        r.bottom_right(),
        r.bottom_left(),
    ])
}

/// Produce the selection overlay (bounding box, resize handles, rotation handle
/// with connector, and/or active marquee) as a [`RenderScene`] of screen-space
/// draw commands.
///
/// * `layout` — present when there is a current selection; drives the bbox and
///   the nine handles.
/// * `marquee` — present while the user is rubber-band selecting; an
///   axis-aligned screen-space rect.
/// * `style` — colors/sizes; use [`OverlayStyle::default`] for Excalidraw-like
///   appearance.
///
/// Passing `None`/`None` yields an empty scene. The commands are emitted in
/// paint order: marquee (bottom), then bbox, then rotation connector + handle,
/// then the resize handles (top, so they stay visible over the border).
pub fn selection_overlay(
    layout: Option<&HandleLayout>,
    marquee: Option<Rect>,
    style: &OverlayStyle,
) -> RenderScene {
    let mut scene = RenderScene::new();

    // --- Marquee (drawn first, underneath any selection UI) ---
    if let Some(rect) = marquee {
        let path = rect_path(rect);
        if !style.marquee_fill.is_transparent() {
            scene.push(DrawCommand::FillPath {
                path: path.clone(),
                paint: Paint::solid(style.marquee_fill),
            });
        }
        scene.push(DrawCommand::StrokePath {
            path,
            stroke: Stroke::with_style(style.marquee_width, StrokeStyle::Dashed),
            paint: Paint::solid(style.marquee_stroke),
        });
        scene.bounds = scene.bounds.union(&rect);
    }

    // --- Selection box + handles ---
    if let Some(layout) = layout {
        let nw = layout.center(Handle::NorthWest);
        let ne = layout.center(Handle::NorthEast);
        let se = layout.center(Handle::SouthEast);
        let sw = layout.center(Handle::SouthWest);

        // Bounding box through the four corner handle centers (NW->NE->SE->SW,
        // closed). Using the corners keeps the box correct under rotation.
        let bbox = Path::polygon(&[nw, ne, se, sw]);
        scene.push(DrawCommand::StrokePath {
            path: bbox,
            stroke: Stroke::solid(style.bbox_width),
            paint: Paint::solid(style.accent),
        });

        // Rotation connector: a line from the top-edge midpoint (North) to the
        // rotation handle center. Drawn before the rotation circle so the circle
        // sits on top.
        let north = layout.center(Handle::North);
        let rot = layout.center(Handle::Rotation);
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[north, rot]),
            stroke: Stroke::solid(style.handle_stroke_width),
            paint: Paint::solid(style.accent),
        });

        // Rotation handle: filled white circle with accent border.
        let circle = circle_path(rot, style.rotation_radius);
        scene.push(DrawCommand::FillPath {
            path: circle.clone(),
            paint: Paint::solid(style.handle_fill),
        });
        scene.push(DrawCommand::StrokePath {
            path: circle,
            stroke: Stroke::solid(style.handle_stroke_width),
            paint: Paint::solid(style.accent),
        });

        // Resize handles: filled white squares with accent border, drawn last so
        // they render over the bbox border.
        for h in Handle::RESIZE {
            let c = layout.center(h);
            let sq = square_path(c, style.handle_size);
            scene.push(DrawCommand::FillPath {
                path: sq.clone(),
                paint: Paint::solid(style.handle_fill),
            });
            scene.push(DrawCommand::StrokePath {
                path: sq,
                stroke: Stroke::solid(style.handle_stroke_width),
                paint: Paint::solid(style.accent),
            });
        }

        // Bounds: union of all handle squares + the rotation circle.
        for h in Handle::RESIZE {
            scene.bounds = scene.bounds.union(&layout.rect(h));
        }
        let r = style.rotation_radius;
        scene.bounds = scene.bounds.union(&Rect::from_min_max(
            rot.x - r,
            rot.y - r,
            rot.x + r,
            rot.y + r,
        ));
    }

    scene
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{PathSegment, Point};
    use crate::interaction::Viewport;
    use std::f64::consts::PI;

    fn layout_for(bbox: Rect, angle: f64) -> HandleLayout {
        HandleLayout::new(bbox, angle, &Viewport::default())
    }

    /// Count filled HANDLE_SIZE-sided squares among the FillPath commands —
    /// i.e. the resize handles, excluding the (larger) marquee fill.
    fn count_handle_squares(scene: &RenderScene) -> usize {
        scene
            .commands
            .iter()
            .filter(|c| match c {
                DrawCommand::FillPath { path, .. } => is_handle_square_path(path),
                _ => false,
            })
            .count()
    }

    fn is_handle_square_path(path: &Path) -> bool {
        // MoveTo + 3 LineTo + Close, no cubics, with side == HANDLE_SIZE.
        if path.segments.len() != 5
            || !matches!(path.segments[0], PathSegment::MoveTo(_))
            || path
                .segments
                .iter()
                .any(|s| matches!(s, PathSegment::CubicTo { .. }))
            || !matches!(path.segments[4], PathSegment::Close)
        {
            return false;
        }
        let pts: Vec<Point> = path
            .segments
            .iter()
            .filter_map(|s| match s {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => Some(*p),
                _ => None,
            })
            .collect();
        let w = (pts[1].x - pts[0].x).abs();
        let h = (pts[2].y - pts[1].y).abs();
        (w - HANDLE_SIZE).abs() < 1e-9 && (h - HANDLE_SIZE).abs() < 1e-9
    }

    fn has_circle_fill(scene: &RenderScene) -> bool {
        scene.commands.iter().any(|c| match c {
            DrawCommand::FillPath { path, .. } => {
                path.segments
                    .iter()
                    .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
                    .count()
                    == 4
            }
            _ => false,
        })
    }

    #[test]
    fn none_none_is_empty() {
        let scene = selection_overlay(None, None, &OverlayStyle::default());
        assert!(scene.is_empty());
        assert_eq!(scene.len(), 0);
    }

    #[test]
    fn marquee_only_yields_just_marquee() {
        let m = Rect::new(10.0, 20.0, 100.0, 50.0);
        let scene = selection_overlay(None, Some(m), &OverlayStyle::default());
        // One fill (faint) + one dashed stroke.
        assert_eq!(scene.len(), 2);
        // No HANDLE_SIZE squares (the only fill is the larger marquee), no circle.
        assert_eq!(count_handle_squares(&scene), 0);
        assert!(!has_circle_fill(&scene));
        // The stroke is dashed.
        let dashed = scene.commands.iter().any(|c| {
            matches!(
                c,
                DrawCommand::StrokePath { stroke, .. } if !stroke.dash.is_empty()
            )
        });
        assert!(dashed, "marquee border must be dashed");
        // Bounds cover the marquee.
        assert!(scene.bounds.contains_rect(&m));
    }

    #[test]
    fn marquee_with_transparent_fill_skips_fill() {
        let m = Rect::new(0.0, 0.0, 10.0, 10.0);
        let style = OverlayStyle {
            marquee_fill: Color::TRANSPARENT,
            ..OverlayStyle::default()
        };
        let scene = selection_overlay(None, Some(m), &style);
        // Only the dashed border remains.
        assert_eq!(scene.len(), 1);
        assert!(matches!(scene.commands[0], DrawCommand::StrokePath { .. }));
    }

    #[test]
    fn layout_yields_bbox_handles_rotation_and_connector() {
        let bbox = Rect::new(100.0, 100.0, 200.0, 100.0);
        let layout = layout_for(bbox, 0.0);
        let scene = selection_overlay(Some(&layout), None, &OverlayStyle::default());

        // Exactly 8 filled handle squares.
        assert_eq!(count_handle_squares(&scene), 8);
        // A rotation circle fill (4 cubics) is present.
        assert!(has_circle_fill(&scene));

        // The first command is the bbox: a closed polygon of 4 corners.
        match &scene.commands[0] {
            DrawCommand::StrokePath { path, .. } => {
                // MoveTo + 3 LineTo + Close == 5 segments through the 4 corners.
                assert_eq!(path.segments.len(), 5);
                assert!(matches!(path.segments[0], PathSegment::MoveTo(_)));
                assert!(matches!(path.segments.last().unwrap(), PathSegment::Close));
            }
            other => panic!("expected bbox stroke first, got {other:?}"),
        }

        // The bbox path passes through the four corner handle centers.
        let nw = layout.center(Handle::NorthWest);
        if let DrawCommand::StrokePath { path, .. } = &scene.commands[0] {
            assert_eq!(path.segments[0], PathSegment::MoveTo(nw));
        }

        // A connector polyline from North center to Rotation center exists.
        let north = layout.center(Handle::North);
        let rot = layout.center(Handle::Rotation);
        let has_connector = scene.commands.iter().any(|c| match c {
            DrawCommand::StrokePath { path, .. } => {
                path.segments.len() == 2
                    && path.segments[0] == PathSegment::MoveTo(north)
                    && path.segments[1] == PathSegment::LineTo(rot)
            }
            _ => false,
        });
        assert!(has_connector, "rotation connector line missing");
    }

    #[test]
    fn command_count_with_full_layout() {
        let layout = layout_for(Rect::new(0.0, 0.0, 50.0, 50.0), 0.0);
        let scene = selection_overlay(Some(&layout), None, &OverlayStyle::default());
        // 1 bbox stroke + 1 connector + (circle fill + circle stroke) +
        // 8 * (square fill + square stroke) = 1 + 1 + 2 + 16 = 20.
        assert_eq!(scene.len(), 20);
    }

    #[test]
    fn both_marquee_and_layout() {
        let layout = layout_for(Rect::new(0.0, 0.0, 50.0, 50.0), 0.0);
        let m = Rect::new(-200.0, -200.0, 20.0, 20.0);
        let scene = selection_overlay(Some(&layout), Some(m), &OverlayStyle::default());
        // marquee fill+stroke (2) + the 20 selection commands.
        assert_eq!(scene.len(), 22);
        // Marquee drawn first (bottom).
        assert!(matches!(scene.commands[0], DrawCommand::FillPath { .. }));
        // Bounds cover both the marquee and the handles.
        assert!(scene.bounds.contains_rect(&m));
        assert!(scene.bounds.contains_rect(&layout.rect(Handle::SouthEast)));
    }

    #[test]
    fn handle_squares_centered_on_handle_centers() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 60.0);
        let layout = layout_for(bbox, 0.0);
        let scene = selection_overlay(Some(&layout), None, &OverlayStyle::default());
        // For each resize handle, a filled square whose 4 corners are centered
        // on the handle center must exist.
        for h in Handle::RESIZE {
            let c = layout.center(h);
            let expected = square_path(c, HANDLE_SIZE);
            let found = scene.commands.iter().any(|cmd| {
                matches!(
                    cmd,
                    DrawCommand::FillPath { path, .. } if *path == expected
                )
            });
            assert!(found, "missing square for {h:?} at {c:?}");
        }
    }

    #[test]
    fn rotation_circle_centered_and_radius_correct() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let layout = layout_for(bbox, 0.0);
        let style = OverlayStyle::default();
        let scene = selection_overlay(Some(&layout), None, &style);
        let rot = layout.center(Handle::Rotation);

        // Find the circle fill and check its rightmost point is rot.x + radius.
        let circle = scene.commands.iter().find_map(|c| match c {
            DrawCommand::FillPath { path, .. }
                if path
                    .segments
                    .iter()
                    .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
                    .count()
                    == 4 =>
            {
                Some(path)
            }
            _ => None,
        });
        let circle = circle.expect("rotation circle present");
        match circle.segments[0] {
            PathSegment::MoveTo(p) => {
                assert!((p.x - (rot.x + style.rotation_radius)).abs() < 1e-9);
                assert!((p.y - rot.y).abs() < 1e-9);
            }
            _ => panic!("circle must start with MoveTo"),
        }
    }

    #[test]
    fn rotated_bbox_uses_rotated_corners() {
        // A rotated selection: bbox path corners must equal the rotated handle
        // centers, not the axis-aligned rect corners.
        let bbox = Rect::new(0.0, 0.0, 100.0, 40.0);
        let layout = layout_for(bbox, PI / 4.0);
        let scene = selection_overlay(Some(&layout), None, &OverlayStyle::default());
        let nw = layout.center(Handle::NorthWest);
        let ne = layout.center(Handle::NorthEast);
        let se = layout.center(Handle::SouthEast);
        let sw = layout.center(Handle::SouthWest);
        match &scene.commands[0] {
            DrawCommand::StrokePath { path, .. } => {
                assert_eq!(path.segments[0], PathSegment::MoveTo(nw));
                assert_eq!(path.segments[1], PathSegment::LineTo(ne));
                assert_eq!(path.segments[2], PathSegment::LineTo(se));
                assert_eq!(path.segments[3], PathSegment::LineTo(sw));
            }
            other => panic!("expected bbox first, got {other:?}"),
        }
    }

    #[test]
    fn circle_path_four_arcs_closed() {
        let p = circle_path(Point::new(5.0, 5.0), 3.0);
        assert!(matches!(p.segments[0], PathSegment::MoveTo(_)));
        assert_eq!(
            p.segments
                .iter()
                .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
                .count(),
            4
        );
        assert_eq!(*p.segments.last().unwrap(), PathSegment::Close);
    }

    #[test]
    fn default_style_is_excalidraw_blue() {
        let s = OverlayStyle::default();
        assert_eq!(s.accent, Color::rgb(0x42, 0x63, 0xeb));
        assert_eq!(s.handle_fill, Color::WHITE);
        assert_eq!(s.handle_size, HANDLE_SIZE);
    }
}
