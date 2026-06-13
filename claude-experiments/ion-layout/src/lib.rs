// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Layered-graph layout ported to Rust from iongraph's generic-layout
// (essence.ts) by Ben Visness — https://github.com/mozilla-spidermonkey/iongraph
// See NOTICE.md for attribution details.
//
// This file is the C ABI surface consumed by the Graphviz plugin
// (plugin/gvplugin_ion.c). The actual layout algorithm lives in core.rs.

pub mod core;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IonPoint {
    pub x: f64,
    pub y: f64,
}

/// Per-edge route descriptor. `offset` indexes into the shared points buffer
/// returned via `route_points`; the route occupies `point_count` consecutive
/// points (cubic bezier control points, 3k+1 of them). `point_count == 0`
/// means the edge has no drawable route.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IonRoute {
    pub offset: usize,
    pub point_count: usize,
    pub arrow_tip: IonPoint,
}

/// Compute the layout for a graph handed over from the C plugin.
///
/// `widths`/`heights` are input+output: the caller passes label sizes in and
/// receives the (clamped) cell sizes the layout actually used, so the plugin
/// can size the rendered node boxes to match.
///
/// `routes` receives one descriptor per input edge, in input order. All route
/// points live in a single buffer returned through `route_points` /
/// `route_points_len`, which the caller must release with
/// `ion_layout_free_points` after copying.
///
/// # Safety
/// All pointers must be valid for the given counts (or null where allowed:
/// loop metadata arrays, route outputs, and the graph dimensions).
#[no_mangle]
pub unsafe extern "C" fn ion_layout_compute(
    node_count: usize,
    widths: *mut f64,
    heights: *mut f64,
    edge_count: usize,
    tails: *const usize,
    heads: *const usize,
    loop_depths: *const i32,
    loop_headers: *const u8,
    backedges: *const u8,
    orientation: u32,
    positions: *mut IonPoint,
    routes: *mut IonRoute,
    route_points: *mut *mut IonPoint,
    route_points_len: *mut usize,
    graph_width: *mut f64,
    graph_height: *mut f64,
) -> i32 {
    if !route_points.is_null() {
        *route_points = std::ptr::null_mut();
    }
    if !route_points_len.is_null() {
        *route_points_len = 0;
    }
    if node_count == 0 {
        if !graph_width.is_null() {
            *graph_width = 0.0;
        }
        if !graph_height.is_null() {
            *graph_height = 0.0;
        }
        return 0;
    }
    if widths.is_null() || heights.is_null() || positions.is_null() {
        return -1;
    }
    if edge_count > 0 && (tails.is_null() || heads.is_null()) {
        return -1;
    }

    let widths = std::slice::from_raw_parts_mut(widths, node_count);
    let heights = std::slice::from_raw_parts_mut(heights, node_count);
    let tail_ids = std::slice::from_raw_parts(tails, edge_count);
    let head_ids = std::slice::from_raw_parts(heads, edge_count);
    let depths = if loop_depths.is_null() { &[][..] } else { std::slice::from_raw_parts(loop_depths, node_count) };
    let headers = if loop_headers.is_null() { &[][..] } else { std::slice::from_raw_parts(loop_headers, node_count) };
    let backs = if backedges.is_null() { &[][..] } else { std::slice::from_raw_parts(backedges, node_count) };
    let positions = std::slice::from_raw_parts_mut(positions, node_count);
    let routes = if routes.is_null() { &mut [][..] } else { std::slice::from_raw_parts_mut(routes, edge_count) };

    let nodes: Vec<core::NodeSpec> = (0..node_count)
        .map(|i| core::NodeSpec {
            width: widths[i],
            height: heights[i],
            loop_depth: depths.get(i).copied().unwrap_or(0),
            loop_header: headers.get(i).copied().unwrap_or(0) != 0,
            backedge: backs.get(i).copied().unwrap_or(0) != 0,
        })
        .collect();
    let edges: Vec<(usize, usize)> = tail_ids.iter().zip(head_ids).map(|(&t, &h)| (t, h)).collect();

    let orient = match orientation {
        1 => core::Orientation::LeftToRight,
        2 => core::Orientation::BottomToTop,
        3 => core::Orientation::RightToLeft,
        _ => core::Orientation::TopToBottom,
    };
    let result = core::layout_oriented(&nodes, &edges, orient);

    for i in 0..node_count {
        positions[i] = result.positions[i];
        let (w, h) = result.node_sizes[i];
        if w > 0.0 {
            widths[i] = w;
        }
        if h > 0.0 {
            heights[i] = h;
        }
    }

    if !routes.is_empty() && !route_points.is_null() && !route_points_len.is_null() {
        let total: usize = result.routes.iter().map(|r| r.points.len()).sum();
        let mut buf: Vec<IonPoint> = Vec::with_capacity(total);
        for (i, route) in result.routes.iter().enumerate().take(routes.len()) {
            routes[i] = IonRoute { offset: buf.len(), point_count: route.points.len(), arrow_tip: route.arrow_tip };
            buf.extend_from_slice(&route.points);
        }
        let boxed = buf.into_boxed_slice();
        *route_points_len = boxed.len();
        *route_points = Box::into_raw(boxed) as *mut IonPoint;
    }

    if !graph_width.is_null() {
        *graph_width = result.width;
    }
    if !graph_height.is_null() {
        *graph_height = result.height;
    }
    0
}

/// Release a points buffer returned by `ion_layout_compute`.
///
/// # Safety
/// `points`/`len` must be exactly the values returned through
/// `route_points`/`route_points_len`, and must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn ion_layout_free_points(points: *mut IonPoint, len: usize) {
    if points.is_null() || len == 0 {
        return;
    }
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(points, len)));
}
