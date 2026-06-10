// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Property-based tests for the layout core, ported from iongraph's
// generic-layout/test.ts and extended with route-level invariants.

use ion_layout::core::{layout, EdgeKind, LayoutResult, NodeSpec, CFG};
use ion_layout::IonPoint;

// ============================================================================
// Graph builder + generators
// ============================================================================

#[derive(Clone, Debug, Default)]
struct G {
    nodes: Vec<NodeSpec>,
    edges: Vec<(usize, usize)>,
}

impl G {
    fn node(&mut self, w: f64, h: f64) -> usize {
        self.nodes.push(NodeSpec { width: w, height: h, ..Default::default() });
        self.nodes.len() - 1
    }
    fn loop_node(&mut self, w: f64, h: f64, depth: i32, header: bool, backedge: bool) -> usize {
        self.nodes.push(NodeSpec { width: w, height: h, loop_depth: depth, loop_header: header, backedge });
        self.nodes.len() - 1
    }
    fn edge(&mut self, t: usize, h: usize) {
        self.edges.push((t, h));
    }
    fn run(&self) -> LayoutResult {
        layout(&self.nodes, &self.edges)
    }
}

/// Same LCG as test.ts for comparable randomized coverage.
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
    fn next(&mut self) -> f64 {
        self.state = (self.state.wrapping_mul(1664525).wrapping_add(1013904223)) & 0x7fffffff;
        self.state as f64 / 0x7fffffff as f64
    }
    fn int(&mut self, min: usize, max: usize) -> usize {
        (self.next() * (max - min + 1) as f64).floor() as usize + min
    }
}

fn chain(n: usize) -> G {
    let mut g = G::default();
    for _ in 0..n {
        g.node(100.0, 40.0);
    }
    for i in 1..n {
        g.edge(i - 1, i);
    }
    g
}

fn diamond() -> G {
    let mut g = G::default();
    for _ in 0..4 {
        g.node(100.0, 40.0);
    }
    g.edge(0, 1);
    g.edge(0, 2);
    g.edge(1, 3);
    g.edge(2, 3);
    g
}

fn single_node() -> G {
    let mut g = G::default();
    g.node(100.0, 40.0);
    g
}

fn wide_fanout(n: usize) -> G {
    let mut g = G::default();
    g.node(120.0, 50.0);
    for _ in 0..n {
        let c = g.node(80.0, 40.0);
        g.edge(0, c);
    }
    g
}

/// entry -> header -> body -> backedge -> header; header -> exit.
/// Explicit ion metadata, like SpiderMonkey provides.
fn simple_loop() -> G {
    let mut g = G::default();
    let entry = g.node(100.0, 40.0);
    let header = g.loop_node(100.0, 40.0, 1, true, false);
    let body = g.loop_node(100.0, 40.0, 1, false, false);
    let back = g.loop_node(80.0, 30.0, 1, false, true);
    let exit = g.node(100.0, 40.0);
    g.edge(entry, header);
    g.edge(header, body);
    g.edge(header, exit);
    g.edge(body, back);
    g.edge(back, header);
    g
}

fn nested_loop() -> G {
    let mut g = G::default();
    let n0 = g.node(100.0, 40.0);
    let n1 = g.loop_node(100.0, 40.0, 1, true, false); // outer header
    let n2 = g.loop_node(100.0, 40.0, 2, true, false); // inner header
    let n3 = g.loop_node(100.0, 40.0, 2, false, false); // inner body
    let n4 = g.loop_node(80.0, 30.0, 2, false, true); // inner backedge
    let n5 = g.loop_node(100.0, 40.0, 1, false, false); // outer body
    let n6 = g.loop_node(80.0, 30.0, 1, false, true); // outer backedge
    let n7 = g.node(100.0, 40.0);
    g.edge(n0, n1);
    g.edge(n1, n2);
    g.edge(n1, n7);
    g.edge(n2, n3);
    g.edge(n2, n5);
    g.edge(n3, n4);
    g.edge(n4, n2);
    g.edge(n5, n6);
    g.edge(n6, n1);
    g
}

/// Loop expressed WITHOUT metadata: the layout must infer it.
/// 0 -> 1 -> 2 -> 1, 1 -> 3
fn inferred_loop() -> G {
    let mut g = G::default();
    for _ in 0..4 {
        g.node(100.0, 40.0);
    }
    g.edge(0, 1);
    g.edge(1, 2);
    g.edge(2, 1);
    g.edge(1, 3);
    g
}

fn self_loop() -> G {
    let mut g = G::default();
    g.node(100.0, 40.0);
    let b = g.node(100.0, 40.0);
    g.node(100.0, 40.0);
    g.edge(0, 1);
    g.edge(b, b);
    g.edge(1, 2);
    g
}

fn multi_edge() -> G {
    let mut g = G::default();
    g.node(100.0, 40.0);
    g.node(100.0, 40.0);
    g.edge(0, 1);
    g.edge(0, 1);
    g
}

fn disconnected() -> G {
    let mut g = G::default();
    for _ in 0..6 {
        g.node(100.0, 40.0);
    }
    g.edge(0, 1);
    g.edge(1, 2);
    g.edge(3, 4);
    g.edge(4, 5);
    g
}

fn pure_cycle() -> G {
    let mut g = G::default();
    g.node(100.0, 40.0);
    g.node(100.0, 40.0);
    g.edge(0, 1);
    g.edge(1, 0);
    g
}

fn skip_edge() -> G {
    let mut g = G::default();
    for _ in 0..4 {
        g.node(100.0, 40.0);
    }
    g.edge(0, 1);
    g.edge(0, 3);
    g.edge(1, 2);
    g.edge(2, 3);
    g
}

fn random_dag(seed: u64, node_count: usize) -> G {
    let mut rng = Rng::new(seed);
    let mut g = G::default();
    for _ in 0..node_count {
        let w = rng.int(60, 200) as f64;
        let h = rng.int(30, 100) as f64;
        g.node(w, h);
    }
    let mut has_pred = vec![false; node_count];
    for i in 0..node_count {
        let max_succs = (node_count - i - 1).min(3);
        let num_succs = rng.int(0, max_succs);
        let mut candidates: Vec<usize> = (i + 1..node_count).collect();
        for _ in 0..num_succs {
            if candidates.is_empty() {
                break;
            }
            let idx = rng.int(0, candidates.len() - 1);
            let succ = candidates.remove(idx);
            g.edge(i, succ);
            has_pred[succ] = true;
        }
    }
    for i in 1..node_count {
        if !has_pred[i] {
            let pred = rng.int(0, i - 1);
            g.edge(pred, i);
            has_pred[i] = true;
        }
    }
    g
}

fn random_loop_graph(seed: u64) -> G {
    let mut rng = Rng::new(seed);
    let body_size = rng.int(1, 4);
    let mut g = G::default();
    let entry = g.node(rng.int(60, 150) as f64, rng.int(30, 80) as f64);
    let header = g.loop_node(rng.int(60, 150) as f64, rng.int(30, 80) as f64, 1, true, false);
    g.edge(entry, header);
    let mut prev = header;
    for _ in 0..body_size {
        let b = g.loop_node(rng.int(60, 150) as f64, rng.int(30, 80) as f64, 1, false, false);
        g.edge(prev, b);
        prev = b;
    }
    let back = g.loop_node(rng.int(40, 80) as f64, rng.int(20, 40) as f64, 1, false, true);
    g.edge(prev, back);
    g.edge(back, header);
    let exit = g.node(rng.int(60, 150) as f64, rng.int(30, 80) as f64);
    g.edge(header, exit);
    g
}

/// Random DAG plus random metadata-free back edges (inference path).
fn random_cfg(seed: u64, node_count: usize, back_edges: usize) -> G {
    let mut g = random_dag(seed, node_count);
    let mut rng = Rng::new(seed.wrapping_add(7_777));
    for _ in 0..back_edges {
        let j = rng.int(1, node_count - 1);
        let i = rng.int(0, j - 1);
        g.edge(j, i);
    }
    g
}

// ============================================================================
// Property checks
// ============================================================================

struct NodeBox {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

fn node_boxes(g: &G, r: &LayoutResult) -> Vec<NodeBox> {
    (0..g.nodes.len())
        .map(|i| {
            let (w, h) = r.node_sizes[i];
            let p = r.positions[i];
            NodeBox { left: p.x - w / 2.0, top: p.y - h / 2.0, right: p.x + w / 2.0, bottom: p.y + h / 2.0 }
        })
        .collect()
}

const EPS: f64 = 1e-6;

fn check_all_nodes_present(g: &G, r: &LayoutResult) {
    assert_eq!(r.positions.len(), g.nodes.len(), "positions count");
    assert_eq!(r.node_layers.len(), g.nodes.len(), "layers count");
    assert_eq!(r.routes.len(), g.edges.len(), "routes count must equal input edge count");
    assert_eq!(r.edge_kinds.len(), g.edges.len(), "edge kinds count");
    for (i, &(w, h)) in r.node_sizes.iter().enumerate() {
        assert!(w > 0.0 && h > 0.0, "node {i} has degenerate size {w}x{h}");
    }
}

fn check_no_overlap(g: &G, r: &LayoutResult) {
    let boxes = node_boxes(g, r);
    for a in 0..boxes.len() {
        for b in (a + 1)..boxes.len() {
            if r.node_layers[a] != r.node_layers[b] {
                continue;
            }
            let (l, rr) = if boxes[a].left <= boxes[b].left { (&boxes[a], &boxes[b]) } else { (&boxes[b], &boxes[a]) };
            let gap = rr.left - l.right;
            assert!(
                gap >= CFG.gap - 1.0,
                "layer {}: nodes {a} and {b} overlap or too close (gap={gap:.1}, need {})",
                r.node_layers[a],
                CFG.gap
            );
        }
    }
}

fn check_layer_monotonicity(g: &G, r: &LayoutResult) {
    for (e, &(t, h)) in g.edges.iter().enumerate() {
        if r.edge_kinds[e] != EdgeKind::Forward {
            continue;
        }
        if g.nodes[t].backedge || g.nodes[h].backedge {
            continue;
        }
        assert!(
            r.node_layers[h] > r.node_layers[t],
            "layer monotonicity violated: edge {t}(layer={}) -> {h}(layer={})",
            r.node_layers[t],
            r.node_layers[h]
        );
    }
}

fn check_backedge_layers(g: &G, r: &LayoutResult) {
    for (i, n) in g.nodes.iter().enumerate() {
        if !n.backedge {
            continue;
        }
        let succs: Vec<usize> = g.edges.iter().filter(|&&(t, _)| t == i).map(|&(_, h)| h).collect();
        assert_eq!(succs.len(), 1, "backedge {i} must have exactly 1 successor");
        assert_eq!(
            r.node_layers[i], r.node_layers[succs[0]],
            "backedge {i} should be on same layer as header {}",
            succs[0]
        );
    }
}

fn check_positive_coordinates(g: &G, r: &LayoutResult) {
    for (i, b) in node_boxes(g, r).iter().enumerate() {
        assert!(b.left >= -EPS, "node {i} has negative left {}", b.left);
        assert!(b.top >= -EPS, "node {i} has negative top {}", b.top);
    }
}

fn check_same_layer_same_top(g: &G, r: &LayoutResult) {
    let boxes = node_boxes(g, r);
    let mut top_by_layer: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for i in 0..boxes.len() {
        if g.nodes[i].backedge {
            continue; // backedge nodes sit at header layer but keep their own height
        }
        let layer = r.node_layers[i];
        let top = boxes[i].top;
        if let Some(&existing) = top_by_layer.get(&layer) {
            assert!(
                (existing - top).abs() < 0.01,
                "nodes on layer {layer} have different top y: {existing} vs {top}"
            );
        } else {
            top_by_layer.insert(layer, top);
        }
    }
}

fn check_layer_y_ordering(g: &G, r: &LayoutResult) {
    let boxes = node_boxes(g, r);
    let mut top_by_layer: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for i in 0..boxes.len() {
        top_by_layer.entry(r.node_layers[i]).or_insert(boxes[i].top);
    }
    let mut layers: Vec<(usize, f64)> = top_by_layer.into_iter().collect();
    layers.sort_by_key(|&(l, _)| l);
    for w in layers.windows(2) {
        assert!(
            w[1].1 > w[0].1,
            "layer {} (y={}) should be below layer {} (y={})",
            w[1].0,
            w[1].1,
            w[0].0,
            w[0].1
        );
    }
}

fn check_bounds(g: &G, r: &LayoutResult) {
    for (i, b) in node_boxes(g, r).iter().enumerate() {
        assert!(b.right <= r.width + EPS, "node {i} exceeds graph width: {} > {}", b.right, r.width);
        assert!(b.bottom <= r.height + EPS, "node {i} exceeds graph height: {} > {}", b.bottom, r.height);
    }
    for (e, route) in r.routes.iter().enumerate() {
        for p in &route.points {
            assert!(p.x >= -EPS && p.x <= r.width + EPS, "edge {e} route x {} outside [0, {}]", p.x, r.width);
            assert!(p.y >= -EPS && p.y <= r.height + EPS, "edge {e} route y {} outside [0, {}]", p.y, r.height);
        }
    }
}

fn check_route_shapes(_g: &G, r: &LayoutResult) {
    for (e, route) in r.routes.iter().enumerate() {
        if r.edge_kinds[e] == EdgeKind::Invalid {
            continue;
        }
        assert!(route.points.len() >= 4, "edge {e} ({:?}) has no usable route", r.edge_kinds[e]);
        assert_eq!(
            (route.points.len() - 1) % 3,
            0,
            "edge {e} route has {} points; bezier needs 3k+1",
            route.points.len()
        );
    }
}

/// Route endpoints must touch their nodes: the start point on the tail's
/// boundary (bottom edge for port exits, left/right edge for channel exits)
/// and the arrow tip on the head's boundary.
fn check_route_endpoints(g: &G, r: &LayoutResult) {
    let boxes = node_boxes(g, r);
    let on_boundary = |b: &NodeBox, p: IonPoint, max_port_x: f64| -> bool {
        let on_bottom = (p.y - b.bottom).abs() < EPS && p.x >= b.left - EPS && p.x <= max_port_x + EPS;
        let on_top = (p.y - b.top).abs() < EPS && p.x >= b.left - EPS && p.x <= max_port_x + EPS;
        let on_left = (p.x - b.left).abs() < EPS && p.y >= b.top - EPS && p.y <= b.bottom + EPS;
        let on_right = (p.x - b.right).abs() < EPS && p.y >= b.top - EPS && p.y <= b.bottom + EPS;
        on_bottom || on_top || on_left || on_right
    };
    for (e, &(t, h)) in g.edges.iter().enumerate() {
        if r.edge_kinds[e] == EdgeKind::Invalid {
            continue;
        }
        let route = &r.routes[e];
        // Ports can extend beyond the label box (essence renders them outside),
        // so allow start x out to the last possible port column.
        let out_degree = g.edges.iter().filter(|&&(t2, _)| t2 == t).count();
        let max_port_x = boxes[t].left + CFG.port_start + CFG.port_step * out_degree.saturating_sub(1) as f64;
        let start = route.points[0];
        assert!(
            on_boundary(&boxes[t], start, max_port_x.max(boxes[t].right)),
            "edge {e} ({t}->{h}, {:?}): start ({:.1},{:.1}) not on tail boundary [{:.1},{:.1}..{:.1},{:.1}]",
            r.edge_kinds[e],
            start.x,
            start.y,
            boxes[t].left,
            boxes[t].top,
            boxes[t].right,
            boxes[t].bottom
        );
        let tip = route.arrow_tip;
        assert!(
            on_boundary(&boxes[h], tip, boxes[h].right),
            "edge {e} ({t}->{h}, {:?}): arrow tip ({:.1},{:.1}) not on head boundary [{:.1},{:.1}..{:.1},{:.1}]",
            r.edge_kinds[e],
            tip.x,
            tip.y,
            boxes[h].left,
            boxes[h].top,
            boxes[h].right,
            boxes[h].bottom
        );
    }
}

fn eval_cubic(p0: IonPoint, p1: IonPoint, p2: IonPoint, p3: IonPoint, t: f64) -> IonPoint {
    let u = 1.0 - t;
    IonPoint {
        x: u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x,
        y: u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y,
    }
}

/// No route may pass through the interior of a node that is not one of its
/// endpoints. Interiors are shrunk by 2px to allow boundary grazing.
fn check_routes_avoid_nodes(g: &G, r: &LayoutResult) {
    let boxes = node_boxes(g, r);
    for (e, &(t, h)) in g.edges.iter().enumerate() {
        if r.edge_kinds[e] == EdgeKind::Invalid {
            continue;
        }
        let route = &r.routes[e];
        let mut samples = Vec::new();
        let mut i = 0;
        while i + 3 < route.points.len() {
            for s in 0..=8 {
                samples.push(eval_cubic(
                    route.points[i],
                    route.points[i + 1],
                    route.points[i + 2],
                    route.points[i + 3],
                    s as f64 / 8.0,
                ));
            }
            i += 3;
        }
        for (n, b) in boxes.iter().enumerate() {
            if n == t || n == h {
                continue;
            }
            let shrink = 2.0;
            for p in &samples {
                let inside = p.x > b.left + shrink && p.x < b.right - shrink && p.y > b.top + shrink && p.y < b.bottom - shrink;
                assert!(
                    !inside,
                    "edge {e} ({t}->{h}, {:?}) passes through node {n} at ({:.1},{:.1}); node box [{:.1},{:.1}..{:.1},{:.1}]",
                    r.edge_kinds[e],
                    p.x,
                    p.y,
                    b.left,
                    b.top,
                    b.right,
                    b.bottom
                );
            }
        }
    }
}

fn check_determinism(g: &G) {
    let a = g.run();
    let b = g.run();
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "layout must be deterministic");
}

fn check_all(g: &G) -> LayoutResult {
    let r = g.run();
    check_all_nodes_present(g, &r);
    check_no_overlap(g, &r);
    check_layer_monotonicity(g, &r);
    check_backedge_layers(g, &r);
    check_positive_coordinates(g, &r);
    check_same_layer_same_top(g, &r);
    check_layer_y_ordering(g, &r);
    check_bounds(g, &r);
    check_route_shapes(g, &r);
    check_route_endpoints(g, &r);
    check_routes_avoid_nodes(g, &r);
    check_determinism(g);
    r
}

// ============================================================================
// Fixed graph tests
// ============================================================================

#[test]
fn test_single_node() {
    check_all(&single_node());
}

#[test]
fn test_chain_2() {
    check_all(&chain(2));
}

#[test]
fn test_chain_5() {
    check_all(&chain(5));
}

#[test]
fn test_chain_20() {
    check_all(&chain(20));
}

#[test]
fn test_diamond() {
    let r = check_all(&diamond());
    assert_eq!(r.node_layers[1], 1);
    assert_eq!(r.node_layers[2], 1);
    assert_eq!(r.node_layers[3], 2);
}

#[test]
fn test_wide_fanout() {
    check_all(&wide_fanout(4));
}

#[test]
fn test_wide_fanout_12() {
    check_all(&wide_fanout(12));
}

#[test]
fn test_simple_loop() {
    check_all(&simple_loop());
}

#[test]
fn test_nested_loop() {
    check_all(&nested_loop());
}

#[test]
fn test_inferred_loop() {
    let g = inferred_loop();
    let r = check_all(&g);
    assert_eq!(r.edge_kinds[2], EdgeKind::Feedback, "2->1 must classify as feedback");
}

#[test]
fn test_self_loop() {
    let g = self_loop();
    let r = check_all(&g);
    assert_eq!(r.edge_kinds[1], EdgeKind::SelfLoop);
}

#[test]
fn test_multi_edge() {
    check_all(&multi_edge());
}

#[test]
fn test_disconnected() {
    check_all(&disconnected());
}

#[test]
fn test_pure_cycle() {
    let g = pure_cycle();
    let r = check_all(&g);
    assert!(r.node_layers[1] > r.node_layers[0]);
}

#[test]
fn test_skip_edge() {
    let g = skip_edge();
    let r = check_all(&g);
    assert_eq!(r.node_layers[3], 3, "node 3 should be pushed to layer 3 by the longer path");
}

#[test]
fn test_very_different_sizes() {
    let mut g = G::default();
    g.node(20.0, 10.0);
    g.node(500.0, 200.0);
    g.node(30.0, 15.0);
    g.edge(0, 1);
    g.edge(1, 2);
    check_all(&g);
}

#[test]
fn test_multiple_roots() {
    let mut g = G::default();
    g.node(100.0, 40.0);
    g.node(100.0, 40.0);
    g.node(100.0, 40.0);
    g.edge(0, 2);
    g.edge(1, 2);
    check_all(&g);
}

#[test]
fn test_empty_graph() {
    let g = G::default();
    let r = g.run();
    assert_eq!(r.positions.len(), 0);
    assert_eq!(r.width, 0.0);
}

// ============================================================================
// Randomized tests
// ============================================================================

#[test]
fn test_random_dags() {
    for seed in 0..300u64 {
        let node_count = 3 + (seed as usize % 30);
        let g = random_dag(seed, node_count);
        let r = g.run();
        check_all_nodes_present(&g, &r);
        check_no_overlap(&g, &r);
        check_layer_monotonicity(&g, &r);
        check_positive_coordinates(&g, &r);
        check_same_layer_same_top(&g, &r);
        check_layer_y_ordering(&g, &r);
        check_bounds(&g, &r);
        check_route_shapes(&g, &r);
        check_route_endpoints(&g, &r);
        check_routes_avoid_nodes(&g, &r);
    }
}

#[test]
fn test_random_loop_graphs() {
    for seed in 0..150u64 {
        let g = random_loop_graph(seed);
        let r = g.run();
        check_all_nodes_present(&g, &r);
        check_no_overlap(&g, &r);
        check_layer_monotonicity(&g, &r);
        check_backedge_layers(&g, &r);
        check_positive_coordinates(&g, &r);
        check_same_layer_same_top(&g, &r);
        check_layer_y_ordering(&g, &r);
        check_bounds(&g, &r);
        check_route_shapes(&g, &r);
        check_route_endpoints(&g, &r);
        check_routes_avoid_nodes(&g, &r);
    }
}

#[test]
fn test_random_cfgs_with_inferred_loops() {
    for seed in 0..200u64 {
        let node_count = 5 + (seed as usize % 25);
        let g = random_cfg(seed, node_count, 1 + seed as usize % 4);
        let r = g.run();
        check_all_nodes_present(&g, &r);
        check_no_overlap(&g, &r);
        check_layer_monotonicity(&g, &r);
        check_positive_coordinates(&g, &r);
        check_same_layer_same_top(&g, &r);
        check_layer_y_ordering(&g, &r);
        check_bounds(&g, &r);
        check_route_shapes(&g, &r);
        check_route_endpoints(&g, &r);
        check_routes_avoid_nodes(&g, &r);
    }
}

// ============================================================================
// Orientation tests
// ============================================================================

#[test]
fn test_orientations_are_exact_transforms() {
    use ion_layout::core::{layout_oriented, Orientation};
    let g = nested_loop();
    // LR runs the TB algorithm on size-swapped nodes; the result must be the
    // exact geometric transpose of that run.
    let swapped: Vec<NodeSpec> = g.nodes.iter().map(|n| NodeSpec { width: n.height, height: n.width, ..*n }).collect();
    let base = layout(&swapped, &g.edges);
    let lr = layout_oriented(&g.nodes, &g.edges, Orientation::LeftToRight);
    assert_eq!(lr.width, base.height);
    assert_eq!(lr.height, base.width);
    for i in 0..g.nodes.len() {
        assert_eq!(lr.positions[i].x, base.positions[i].y, "node {i} x");
        assert_eq!(lr.positions[i].y, base.positions[i].x, "node {i} y");
        assert_eq!(lr.node_sizes[i], (base.node_sizes[i].1, base.node_sizes[i].0), "node {i} size");
    }
    for (e, route) in lr.routes.iter().enumerate() {
        assert_eq!(route.points.len(), base.routes[e].points.len());
        for (p, q) in route.points.iter().zip(&base.routes[e].points) {
            assert_eq!((p.x, p.y), (q.y, q.x), "edge {e} point");
        }
    }

    // BT is TB mirrored along y.
    let tb = layout_oriented(&g.nodes, &g.edges, Orientation::TopToBottom);
    let bt = layout_oriented(&g.nodes, &g.edges, Orientation::BottomToTop);
    for i in 0..g.nodes.len() {
        assert_eq!(bt.positions[i].x, tb.positions[i].x);
        assert!((bt.positions[i].y - (tb.height - tb.positions[i].y)).abs() < 1e-9);
    }
}

#[test]
fn test_lr_layout_invariants() {
    use ion_layout::core::{layout_oriented, Orientation};
    // The transposed layout must still keep node boxes disjoint in render
    // space and routes within bounds.
    for seed in 0..40u64 {
        let g = random_dag(seed, 4 + (seed as usize % 12));
        let r = layout_oriented(&g.nodes, &g.edges, Orientation::LeftToRight);
        let boxes: Vec<(f64, f64, f64, f64)> = (0..g.nodes.len())
            .map(|i| {
                let (w, h) = r.node_sizes[i];
                let p = r.positions[i];
                (p.x - w / 2.0, p.y - h / 2.0, p.x + w / 2.0, p.y + h / 2.0)
            })
            .collect();
        for a in 0..boxes.len() {
            for b in (a + 1)..boxes.len() {
                let ox = boxes[a].2.min(boxes[b].2) - boxes[a].0.max(boxes[b].0);
                let oy = boxes[a].3.min(boxes[b].3) - boxes[a].1.max(boxes[b].1);
                assert!(!(ox > 0.5 && oy > 0.5), "seed {seed}: nodes {a}/{b} overlap in LR layout");
            }
        }
        for route in &r.routes {
            for p in &route.points {
                assert!(p.x >= -1e-6 && p.x <= r.width + 1e-6, "seed {seed}: route x {} out of [0,{}]", p.x, r.width);
                assert!(p.y >= -1e-6 && p.y <= r.height + 1e-6, "seed {seed}: route y {} out of [0,{}]", p.y, r.height);
            }
        }
    }
}
