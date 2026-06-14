//! Selection transform handles: the 8 resize handles + 1 rotation handle drawn
//! around a selection's bounding box, plus hit-testing against them.
//!
//! Reimplemented from Excalidraw's transform-handle logic
//! (`packages/element/src/transformHandles.ts` and
//! `packages/element/src/resizeElements.ts`). We keep the same handle set and the
//! same screen-space sizing (handles are a fixed pixel size regardless of zoom)
//! but express it in idiomatic Rust against this crate's geometry types.
//!
//! Positions are computed in **screen** coordinates because handle size and the
//! rotation-handle offset are pixel quantities the user interacts with directly;
//! callers map the pointer into screen space (or pass the handle rect through the
//! viewport) before testing. The interaction state machine maps the resulting
//! drag back into scene space.

use crate::geometry::{point_rotate_rads, Point, Rect};
use crate::interaction::Viewport;

/// The fixed on-screen size (in CSS-ish pixels) of a square resize handle.
/// Matches Excalidraw's default `transformHandleSizes`.
pub const HANDLE_SIZE: f64 = 8.0;

/// How far above the selection's top edge the rotation handle sits, in screen
/// pixels. Matches Excalidraw's `ROTATION_RESIZE_HANDLE_GAP` (+ handle size).
pub const ROTATION_HANDLE_OFFSET: f64 = 28.0;

/// The nine interactive handles around a selection box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Handle {
    NorthWest,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    /// The single rotation handle, above the top edge.
    Rotation,
}

impl Handle {
    /// The eight resize handles, in clockwise order from the top-left.
    pub const RESIZE: [Handle; 8] = [
        Handle::NorthWest,
        Handle::North,
        Handle::NorthEast,
        Handle::East,
        Handle::SouthEast,
        Handle::South,
        Handle::SouthWest,
        Handle::West,
    ];

    /// All nine handles (resize + rotation).
    pub const ALL: [Handle; 9] = [
        Handle::NorthWest,
        Handle::North,
        Handle::NorthEast,
        Handle::East,
        Handle::SouthEast,
        Handle::South,
        Handle::SouthWest,
        Handle::West,
        Handle::Rotation,
    ];

    pub fn is_corner(self) -> bool {
        matches!(
            self,
            Handle::NorthWest | Handle::NorthEast | Handle::SouthEast | Handle::SouthWest
        )
    }

    pub fn is_rotation(self) -> bool {
        matches!(self, Handle::Rotation)
    }

    /// Does this handle move the box's left edge when dragged?
    pub fn affects_left(self) -> bool {
        matches!(self, Handle::NorthWest | Handle::West | Handle::SouthWest)
    }
    /// Does this handle move the box's right edge when dragged?
    pub fn affects_right(self) -> bool {
        matches!(self, Handle::NorthEast | Handle::East | Handle::SouthEast)
    }
    /// Does this handle move the box's top edge when dragged?
    pub fn affects_top(self) -> bool {
        matches!(self, Handle::NorthWest | Handle::North | Handle::NorthEast)
    }
    /// Does this handle move the box's bottom edge when dragged?
    pub fn affects_bottom(self) -> bool {
        matches!(self, Handle::SouthWest | Handle::South | Handle::SouthEast)
    }
}

/// The set of handles laid out around a selection bounding box.
///
/// The selection box is given in **scene** coordinates (axis-aligned), together
/// with the selection's rotation `angle` (radians, clockwise). Handle centers are
/// returned in **screen** coordinates so a fixed pixel size and rotation offset
/// can be applied independent of zoom.
#[derive(Debug, Clone, Copy)]
pub struct HandleLayout {
    /// Screen-space center of each resize handle, indexed by [`Handle::RESIZE`].
    centers: [Point; 8],
    /// Screen-space center of the rotation handle.
    rotation: Point,
    /// Screen-space pivot (the selection box center) used by rotation.
    pivot: Point,
    /// Half the interactive square side, in screen pixels.
    half: f64,
}

impl HandleLayout {
    /// Compute handle positions for `bbox` (scene coords) rotated by `angle`.
    pub fn new(bbox: Rect, angle: f64, vp: &Viewport) -> Self {
        let to_screen = vp.scene_to_screen();
        // Unrotated screen-space corners/edges, then rotate each around the
        // screen-space center by `angle` (rotation preserves under the uniform
        // scale + translate of the viewport, so applying it in screen space with
        // the same angle is correct for the uniform zoom we use).
        let center_scene = bbox.center();
        let pivot = to_screen.apply(center_scene);

        let raw = |p: Point| point_rotate_rads(to_screen.apply(p), pivot, angle);

        let nw = raw(bbox.top_left());
        let ne = raw(bbox.top_right());
        let se = raw(bbox.bottom_right());
        let sw = raw(bbox.bottom_left());
        let n = raw(Point::new(bbox.center().x, bbox.min_y()));
        let e = raw(Point::new(bbox.max_x(), bbox.center().y));
        let s = raw(Point::new(bbox.center().x, bbox.max_y()));
        let w = raw(Point::new(bbox.min_x(), bbox.center().y));

        // Rotation handle sits a fixed screen offset off the (rotated) top-edge
        // midpoint, along the box's local "up" direction. Local up before
        // rotation is (0, -1); rotate that direction by `angle` and offset `n`
        // (the already-rotated top-edge midpoint) along it.
        let rotation = {
            let (sin, cos) = angle.sin_cos();
            // rotate the unit vector (0, -1) clockwise by `angle`
            let up = Point::new(sin, -cos);
            Point::new(
                n.x + up.x * ROTATION_HANDLE_OFFSET,
                n.y + up.y * ROTATION_HANDLE_OFFSET,
            )
        };

        HandleLayout {
            centers: [nw, n, ne, e, se, s, sw, w],
            rotation,
            pivot,
            half: HANDLE_SIZE / 2.0,
        }
    }

    /// Screen-space center of a handle.
    pub fn center(&self, h: Handle) -> Point {
        match h {
            Handle::Rotation => self.rotation,
            other => {
                let idx = Handle::RESIZE
                    .iter()
                    .position(|x| *x == other)
                    .expect("non-rotation handle is in RESIZE");
                self.centers[idx]
            }
        }
    }

    /// The selection pivot (screen-space box center), used as rotation center.
    pub fn pivot(&self) -> Point {
        self.pivot
    }

    /// Screen-space square that is the interactive region of a resize handle.
    pub fn rect(&self, h: Handle) -> Rect {
        let c = self.center(h);
        Rect::new(c.x - self.half, c.y - self.half, HANDLE_SIZE, HANDLE_SIZE)
    }

    /// Hit-test a screen-space point against all handles. Corners take priority
    /// over edges, and the rotation handle is tested with a circular tolerance.
    /// Returns the matched handle, if any.
    pub fn hit(&self, screen_pt: Point) -> Option<Handle> {
        // Rotation handle: circular hit area of radius `half + slack`.
        let rot_radius = self.half + 2.0;
        if self.rotation.distance(screen_pt) <= rot_radius {
            return Some(Handle::Rotation);
        }
        // Corners first.
        for h in Handle::RESIZE.iter().filter(|h| h.is_corner()) {
            if self.rect(*h).contains(screen_pt) {
                return Some(*h);
            }
        }
        // Then edges.
        for h in Handle::RESIZE.iter().filter(|h| !h.is_corner()) {
            if self.rect(*h).contains(screen_pt) {
                return Some(*h);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn vp() -> Viewport {
        Viewport::default()
    }

    #[test]
    fn handle_positions_unrotated_identity_viewport() {
        let bbox = Rect::new(100.0, 100.0, 200.0, 100.0);
        let layout = HandleLayout::new(bbox, 0.0, &vp());
        assert_eq!(layout.center(Handle::NorthWest), Point::new(100.0, 100.0));
        assert_eq!(layout.center(Handle::SouthEast), Point::new(300.0, 200.0));
        assert_eq!(layout.center(Handle::North), Point::new(200.0, 100.0));
        assert_eq!(layout.center(Handle::East), Point::new(300.0, 150.0));
        // pivot is the box center
        assert_eq!(layout.pivot(), Point::new(200.0, 150.0));
    }

    #[test]
    fn rotation_handle_above_top_when_unrotated() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let layout = HandleLayout::new(bbox, 0.0, &vp());
        let rot = layout.center(Handle::Rotation);
        // directly above the top-edge midpoint
        assert!((rot.x - 50.0).abs() < 1e-9, "x={}", rot.x);
        assert!(
            (rot.y - (0.0 - ROTATION_HANDLE_OFFSET)).abs() < 1e-9,
            "y={}",
            rot.y
        );
    }

    #[test]
    fn hit_corner_takes_priority() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let layout = HandleLayout::new(bbox, 0.0, &vp());
        // exactly on the NW corner center
        assert_eq!(layout.hit(Point::new(0.0, 0.0)), Some(Handle::NorthWest));
        // near the rotation handle
        assert_eq!(
            layout.hit(Point::new(50.0, -ROTATION_HANDLE_OFFSET)),
            Some(Handle::Rotation)
        );
        // empty space
        assert_eq!(layout.hit(Point::new(50.0, 50.0)), None);
    }

    #[test]
    fn zoom_keeps_pixel_handle_size() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        let v = Viewport {
            zoom: 2.0,
            ..Default::default()
        };
        let layout = HandleLayout::new(bbox, 0.0, &v);
        let r = layout.rect(Handle::SouthEast);
        // Handle square is a fixed screen size regardless of zoom.
        assert_eq!(r.width, HANDLE_SIZE);
        assert_eq!(r.height, HANDLE_SIZE);
    }

    #[test]
    fn rotation_handle_follows_angle() {
        let bbox = Rect::new(0.0, 0.0, 100.0, 100.0);
        // Rotate 90deg clockwise: local "up" now points in +x screen direction.
        let layout = HandleLayout::new(bbox, PI / 2.0, &vp());
        let rot = layout.center(Handle::Rotation);
        let pivot = layout.pivot();
        // Rotation handle should be offset roughly horizontally from the box now.
        let dx = rot.x - pivot.x;
        let dy = rot.y - pivot.y;
        assert!(
            dx.abs() > dy.abs(),
            "expected horizontal offset: dx={dx} dy={dy}"
        );
    }

    #[test]
    fn affects_edges_flags() {
        assert!(Handle::NorthWest.affects_left());
        assert!(Handle::NorthWest.affects_top());
        assert!(!Handle::NorthWest.affects_right());
        assert!(Handle::SouthEast.affects_right());
        assert!(Handle::SouthEast.affects_bottom());
        assert!(Handle::North.affects_top());
        assert!(!Handle::North.affects_left());
    }
}
