// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Layered-graph layout ported to Rust from iongraph by Ben Visness —
// https://github.com/mozilla-spidermonkey/iongraph
// This is a faithful port of generic-layout/layout.ts (the real viewer
// algorithm), with a Graphviz-friendly frontend bolted on: edges are
// classified and loop metadata is inferred when absent, so arbitrary DOT
// graphs work. See NOTICE.md for attribution details.
//
// Safe core API. The C ABI wrapper lives in lib.rs; everything here is
// plain Rust so it can be exercised directly by tests and dev binaries.

// Index-based loops are pervasive here because passes mutate node arrays
// while iterating layer index lists; routing helpers carry full layout
// context.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};

use crate::IonPoint;

/// Input description of one node.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeSpec {
    pub width: f64,
    pub height: f64,
    pub loop_depth: i32,
    pub loop_header: bool,
    pub backedge: bool,
}

/// A renderable edge route: cubic bezier control points (3k+1 of them)
/// plus the point the arrowhead should aim at. The curve itself stops
/// short of the arrow tip so the arrowhead has room.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Route {
    pub points: Vec<IonPoint>,
    pub arrow_tip: IonPoint,
}

/// Result of a layout run. `routes` has exactly one entry per input edge,
/// in input order.
#[derive(Clone, Debug, Default)]
pub struct LayoutResult {
    /// Node centers, one per input node.
    pub positions: Vec<IonPoint>,
    /// Clamped node sizes actually used by the layout (w, h).
    pub node_sizes: Vec<(f64, f64)>,
    /// One route per input edge, in input order.
    pub routes: Vec<Route>,
    pub width: f64,
    pub height: f64,
    /// Layer index assigned to each input node (debugging/verification).
    pub node_layers: Vec<usize>,
    /// How each input edge was classified, in input order.
    pub edge_kinds: Vec<EdgeKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Forward,
    Feedback,
    SelfLoop,
    Invalid,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub gap: f64,
    pub port_start: f64,
    pub port_step: f64,
    pub padding: f64,
    pub track_pad: f64,
    pub track_step: f64,
    pub radius: f64,
    pub layout_iterations: usize,
    pub nearly_straight: f64,
    pub nearly_straight_iterations: usize,
}

pub const CFG: Config = Config {
    gap: 44.0,
    port_start: 16.0,
    port_step: 60.0,
    padding: 20.0,
    track_pad: 36.0,
    track_step: 16.0,
    radius: 12.0,
    layout_iterations: 2,
    nearly_straight: 30.0,
    nearly_straight_iterations: 8,
};

const ARROW_INSET: f64 = 10.0;
const HEADER_PUSHDOWN: f64 = 16.0;
/// Only truly degenerate (zero/negative) sizes get clamped, so caller-given
/// sizes pass through byte-exact; multi-port nodes are additionally widened
/// to fit their ports (see MIN_PORT_STEP).
pub const MIN_NODE_W: f64 = 1.0;
pub const MIN_NODE_H: f64 = 1.0;
/// Ports compress to fit inside their node, but never tighter than this;
/// nodes too narrow for their port count get widened instead.
pub const MIN_PORT_STEP: f64 = 24.0;

/// Flow direction, mirroring Graphviz's `rankdir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    #[default]
    TopToBottom,
    LeftToRight,
    BottomToTop,
    RightToLeft,
}

/// Layout honoring a flow direction. The algorithm itself is top-to-bottom;
/// other orientations run it in transposed/mirrored space (node sizes are
/// swapped going in, all geometry is mapped back coming out), so ports land
/// on the sides for horizontal flow and invariants hold in render space.
pub fn layout_oriented(nodes_in: &[NodeSpec], edges_in: &[(usize, usize)], orient: Orientation) -> LayoutResult {
    use Orientation::*;
    let transpose = matches!(orient, LeftToRight | RightToLeft);
    let mirror = matches!(orient, BottomToTop | RightToLeft);

    let specs: Vec<NodeSpec>;
    let nodes = if transpose {
        specs = nodes_in.iter().map(|n| NodeSpec { width: n.height, height: n.width, ..*n }).collect();
        &specs[..]
    } else {
        nodes_in
    };

    let mut r = layout(nodes, edges_in);

    if transpose {
        for p in &mut r.positions {
            std::mem::swap(&mut p.x, &mut p.y);
        }
        for s in &mut r.node_sizes {
            *s = (s.1, s.0);
        }
        for route in &mut r.routes {
            for p in &mut route.points {
                std::mem::swap(&mut p.x, &mut p.y);
            }
            std::mem::swap(&mut route.arrow_tip.x, &mut route.arrow_tip.y);
        }
        std::mem::swap(&mut r.width, &mut r.height);
    }
    if mirror {
        // Mirror along the flow axis: x for horizontal flow, y for vertical.
        if transpose {
            for p in &mut r.positions {
                p.x = r.width - p.x;
            }
            for route in &mut r.routes {
                for p in &mut route.points {
                    p.x = r.width - p.x;
                }
                route.arrow_tip.x = r.width - route.arrow_tip.x;
            }
        } else {
            for p in &mut r.positions {
                p.y = r.height - p.y;
            }
            for route in &mut r.routes {
                for p in &mut route.points {
                    p.y = r.height - p.y;
                }
                route.arrow_tip.y = r.height - route.arrow_tip.y;
            }
        }
    }
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal model (mirrors layout.ts)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct Block {
    w: f64,
    h: f64,
    preds: Vec<usize>,
    succs: Vec<usize>,
    loop_depth: i32,
    is_loop_header: bool,
    is_backedge: bool,
    layer: i64,
    loop_id: i64,
    layout_node: usize,
    loop_height: i64,
    parent_loop: Option<usize>,
    outgoing_edges: Vec<usize>,
    backedge: Option<usize>,
}

const LEFTMOST_DUMMY: u8 = 1 << 0;
const RIGHTMOST_DUMMY: u8 = 1 << 1;
const IMMINENT_BACKEDGE_DUMMY: u8 = 1 << 2;

#[derive(Clone, Debug)]
struct LNode {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    /// Some(block index) for real nodes, None for dummies.
    block: Option<usize>,
    /// For dummies: the destination block this dummy chain leads to.
    dst_block: Option<usize>,
    /// dst slots: for real nodes, indexed by successor position; dummies
    /// have exactly one slot.
    dst: Vec<usize>,
    src: Vec<usize>,
    joints: Vec<f64>,
    flags: u8,
    /// Layer index (into the compressed layer list).
    layer: usize,
    /// Number of output port slots (for port compression).
    ports: usize,
}

const NONE: usize = usize::MAX;

#[derive(Clone, Debug)]
struct EdgeRec {
    tail: usize,
    head: usize,
    kind: EdgeKind,
    /// Slot in the tail block's succs list (NONE for edges that never enter
    /// the layout: self-loops and dropped feedback edges).
    slot: usize,
}

pub fn layout(nodes_in: &[NodeSpec], edges_in: &[(usize, usize)]) -> LayoutResult {
    let node_count = nodes_in.len();
    if node_count == 0 {
        return LayoutResult { routes: vec![Route::default(); edges_in.len()], ..Default::default() };
    }
    let cfg = CFG;

    let has_ion_metadata = nodes_in.iter().any(|n| n.loop_depth != 0 || n.loop_header || n.backedge);

    // ── Frontend: classify edges so arbitrary digraphs become valid input ──
    // (the layout algorithm itself requires a DAG plus marked backedge
    // blocks; we classify forward/feedback by incremental reachability and,
    // when no ion metadata is present, infer loops and synthesize one
    // zero-size backedge block per loop header).
    let mut recs: Vec<EdgeRec> = Vec::with_capacity(edges_in.len());
    let mut forward_adj = vec![Vec::<usize>::new(); node_count];
    for &(tail, head) in edges_in {
        let kind = if tail >= node_count || head >= node_count {
            EdgeKind::Invalid
        } else if tail == head {
            EdgeKind::SelfLoop
        } else if reaches(head, tail, &forward_adj) {
            EdgeKind::Feedback
        } else {
            forward_adj[tail].push(head);
            EdgeKind::Forward
        };
        recs.push(EdgeRec { tail, head, kind, slot: NONE });
    }

    let mut final_headers: Vec<bool> = nodes_in.iter().map(|n| n.loop_header).collect();
    let mut final_depths: Vec<i32> = nodes_in.iter().map(|n| n.loop_depth).collect();
    let mut is_backedge_node: Vec<bool> = nodes_in.iter().map(|n| n.backedge).collect();

    // Demote extra backedge blocks: the algorithm supports exactly one
    // backedge predecessor per header. Keep the first per header.
    if has_ion_metadata {
        let mut seen_backedge_for: HashSet<usize> = HashSet::new();
        for i in 0..node_count {
            if !is_backedge_node[i] {
                continue;
            }
            let header = recs.iter().find(|r| r.tail == i && r.kind != EdgeKind::Invalid).map(|r| r.head);
            if let Some(h) = header {
                if !seen_backedge_for.insert(h) {
                    is_backedge_node[i] = false;
                }
            } else {
                is_backedge_node[i] = false; // backedge with no outgoing edge
            }
        }
        // Headers whose backedge went missing entirely get demoted too.
        for h in 0..node_count {
            if final_headers[h] {
                let has_back = recs.iter().any(|r| r.kind != EdgeKind::Invalid && r.head == h && is_backedge_node[r.tail]);
                if !has_back {
                    final_headers[h] = false;
                }
            }
        }
    }

    // One inferred loop per HEADER (a header can have many feedback edges —
    // think multiple `continue` statements). All of a header's feedback edges
    // share one synthesized backedge block; the natural-loop body is the
    // union over all its back-edge tails.
    let mut inferred_loops: Vec<(usize, Vec<usize>)> = Vec::new();
    if !has_ion_metadata {
        let mut preds = vec![Vec::<usize>::new(); node_count];
        for t in 0..node_count {
            for &h in &forward_adj[t] {
                preds[h].push(t);
            }
        }
        for rec in &recs {
            if rec.kind != EdgeKind::Feedback {
                continue;
            }
            if let Some(entry) = inferred_loops.iter_mut().find(|(h, _)| *h == rec.head) {
                entry.1.push(rec.tail);
            } else {
                inferred_loops.push((rec.head, vec![rec.tail]));
            }
        }
        for (h, tails) in &inferred_loops {
            final_headers[*h] = true;
            let mut body = HashSet::new();
            body.insert(*h);
            let mut stk = Vec::new();
            for &t in tails {
                if body.insert(t) {
                    stk.push(t);
                }
            }
            while let Some(n) = stk.pop() {
                for &p in &preds[n] {
                    if body.insert(p) {
                        stk.push(p);
                    }
                }
            }
            for &n in &body {
                if n < node_count {
                    final_depths[n] += 1;
                }
            }
        }
    }

    let synth_count = inferred_loops.len();
    let total_count = node_count + synth_count;
    let mut synth_for_header: HashMap<usize, usize> = HashMap::new();
    for (i, &(h, _)) in inferred_loops.iter().enumerate() {
        synth_for_header.insert(h, node_count + i);
    }

    // Per-block successor lists. Self-loops and feedback edges that don't go
    // through a backedge block are EXCLUDED — this is what guarantees the
    // succ graph (minus backedge blocks) is acyclic, even on inputs whose
    // metadata is wrong (the upstream algorithm hangs on those).
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); total_count];
    let mut succ_recs: Vec<Vec<usize>> = vec![Vec::new(); total_count];
    for rec_idx in 0..recs.len() {
        let rec = &mut recs[rec_idx];
        if rec.kind == EdgeKind::Invalid || rec.kind == EdgeKind::SelfLoop {
            continue;
        }
        let (tail, head) = (rec.tail, rec.head);
        match rec.kind {
            EdgeKind::Feedback => {
                if let Some(&synth_id) = synth_for_header.get(&head) {
                    rec.slot = succs[tail].len();
                    succs[tail].push(synth_id);
                    succ_recs[tail].push(rec_idx);
                } else if is_backedge_node[tail] || is_backedge_node[head] {
                    rec.slot = succs[tail].len();
                    succs[tail].push(head);
                    succ_recs[tail].push(rec_idx);
                }
                // else: dropped from the layout; routed via fallback.
            }
            EdgeKind::Forward => {
                rec.slot = succs[tail].len();
                succs[tail].push(head);
                succ_recs[tail].push(rec_idx);
            }
            _ => unreachable!(),
        }
    }
    for (i, &(h, _)) in inferred_loops.iter().enumerate() {
        succs[node_count + i].push(h);
        succ_recs[node_count + i].push(NONE);
    }

    // Backedge blocks must have exactly one successor (their header) in the
    // layout; extra succs on an explicit backedge block get dropped from the
    // grid (routed via fallback).
    for b in 0..node_count {
        let backish = is_backedge_node[b];
        if backish && succs[b].len() > 1 {
            for k in (1..succs[b].len()).rev() {
                let r = succ_recs[b][k];
                if r != NONE {
                    recs[r].slot = NONE;
                }
                succs[b].remove(k);
                succ_recs[b].remove(k);
            }
        }
        if backish && succs[b].is_empty() {
            is_backedge_node[b] = false;
        }
    }

    // ── Build blocks ──
    let mut blocks: Vec<Block> = (0..total_count)
        .map(|i| {
            let (w, h, depth, header, backedge) = if i < node_count {
                let ports = succs[i].len();
                let min_port_w = if ports > 1 { 2.0 * cfg.port_start + MIN_PORT_STEP * (ports - 1) as f64 } else { 0.0 };
                (
                    nodes_in[i].width.max(MIN_NODE_W).max(min_port_w),
                    nodes_in[i].height.max(MIN_NODE_H),
                    final_depths[i],
                    final_headers[i],
                    is_backedge_node[i],
                )
            } else {
                let header = inferred_loops[i - node_count].0;
                (0.0, 0.0, final_depths[header], false, true)
            };
            Block {
                w,
                h,
                preds: Vec::new(),
                succs: succs[i].clone(),
                loop_depth: depth,
                is_loop_header: header,
                is_backedge: backedge,
                layer: -1,
                loop_id: -1,
                layout_node: NONE,
                loop_height: 0,
                parent_loop: None,
                outgoing_edges: Vec::new(),
                backedge: None,
            }
        })
        .collect();
    for b in 0..total_count {
        for k in 0..blocks[b].succs.len() {
            let s = blocks[b].succs[k];
            blocks[s].preds.push(b);
        }
    }
    for h in 0..total_count {
        if blocks[h].is_loop_header {
            blocks[h].backedge = blocks[h].preds.iter().copied().find(|&p| blocks[p].is_backedge);
            if blocks[h].backedge.is_none() {
                blocks[h].is_loop_header = false;
            }
        }
    }

    // ── Roots ──
    let mut roots: Vec<usize> = (0..total_count).filter(|&b| blocks[b].preds.is_empty()).collect();
    if roots.is_empty() {
        // Pure cycles with no entry: upstream throws; we degrade by treating
        // the lowest-index block as a root so every graph still lays out.
        roots.push(0);
    }

    for &r in &roots {
        find_loops(&mut blocks, r);
    }
    for &r in &roots {
        assign_layers(&mut blocks, r);
    }
    // Anything unreachable from the roots (possible in degraded inputs)
    // still needs a layer.
    loop {
        let Some(next) = (0..total_count).find(|&b| blocks[b].layer < 0 && !blocks[b].is_backedge) else { break };
        find_loops(&mut blocks, next);
        assign_layers(&mut blocks, next);
    }
    for b in 0..total_count {
        if blocks[b].layer < 0 {
            blocks[b].layer = 0;
        }
    }
    // Degraded inputs (side entries into inferred loop bodies) can leave
    // orphaned deferred loop exits at stale shallow layers, breaking layer
    // monotonicity. Enforce it with a bounded fixpoint over the same visit
    // machinery; on well-formed input this is a no-op.
    let mut budget = total_count.saturating_mul(total_count) + 16;
    'fixpoint: loop {
        let mut fixed_any = false;
        for b in 0..total_count {
            if blocks[b].is_backedge {
                continue;
            }
            for k in 0..blocks[b].succs.len() {
                let s = blocks[b].succs[k];
                if blocks[s].is_backedge {
                    continue;
                }
                if blocks[s].layer <= blocks[b].layer {
                    let at = blocks[b].layer + 1;
                    assign_layers_from(&mut blocks, s, at);
                    fixed_any = true;
                }
            }
        }
        if !fixed_any {
            break 'fixpoint;
        }
        budget = budget.saturating_sub(1);
        if budget == 0 {
            break 'fixpoint;
        }
    }

    // Backedge blocks pin to their header's layer. assign_layers does this at
    // visit time, but the header can be pushed deeper afterwards without the
    // backedge being revisited; re-pin so the value is always final.
    for b in 0..total_count {
        if blocks[b].is_backedge {
            if let Some(&succ) = blocks[b].succs.first() {
                blocks[b].layer = blocks[succ].layer;
            }
        }
    }

    // Compress sparse layer numbers into contiguous indices.
    let mut layer_numbers: Vec<i64> = blocks.iter().map(|b| b.layer).collect::<HashSet<_>>().into_iter().collect();
    layer_numbers.sort();
    let layer_index: HashMap<i64, usize> = layer_numbers.iter().enumerate().map(|(i, &l)| (l, i)).collect();
    let layer_count = layer_numbers.len();

    // ── Layout graph construction (makeLayoutNodes) ──
    let (mut lnodes, mut layers) = make_layout_nodes(&mut blocks, &layer_index, layer_count, &cfg);

    // ── Horizontal straightening ──
    straighten_edges(&mut lnodes, &mut layers, &blocks, &cfg);

    // The left-dummy compaction has no left floor; shift the whole drawing
    // right so nothing renders at negative coordinates.
    let min_x = lnodes.iter().map(|n| n.x).fold(f64::INFINITY, f64::min);
    if min_x.is_finite() && min_x < cfg.padding {
        let dx = cfg.padding - min_x;
        for n in lnodes.iter_mut() {
            n.x += dx;
        }
    }

    if std::env::var("ION_DEBUG_LNODES").is_ok() {
        for (i, n) in lnodes.iter().enumerate() {
            eprintln!("lnode {i}: layer={} block={:?} dst_block={:?} flags={} x={:.0} dst={:?} src={:?}", n.layer, n.block, n.dst_block, n.flags, n.x, n.dst.iter().map(|&d| if d==usize::MAX {-1i64} else {d as i64}).collect::<Vec<_>>(), n.src);
        }
        for (b, blk) in blocks.iter().enumerate() {
            eprintln!("block {b}: layer={} loop_id={} header={} backedge={} back={:?} lnode={}", blk.layer, blk.loop_id, blk.is_loop_header, blk.is_backedge, blk.backedge, blk.layout_node as i64);
        }
    }

    // ── Joints / tracks ──
    let track_h = compute_joints(&mut lnodes, &layers, &blocks, &cfg);

    // ── Vertical placement ──
    let layer_h = verticalize(&mut lnodes, &layers, &track_h, &cfg);

    // ── Extract results ──
    let mut width: f64 = 0.0;
    let mut height: f64 = 0.0;
    for n in &lnodes {
        width = width.max(n.x + n.w + cfg.padding);
        height = height.max(n.y + n.h + cfg.padding);
    }

    let mut positions = vec![IonPoint::default(); node_count];
    let mut node_sizes = vec![(0.0, 0.0); node_count];
    let mut node_layers = vec![0usize; node_count];
    for b in 0..node_count {
        let ln = blocks[b].layout_node;
        if ln == NONE {
            continue;
        }
        let n = &lnodes[ln];
        positions[b] = IonPoint { x: n.x + n.w / 2.0, y: n.y + n.h / 2.0 };
        node_sizes[b] = (n.w, n.h);
        node_layers[b] = n.layer;
    }

    let mut routes = vec![Route::default(); recs.len()];
    for (i, rec) in recs.iter().enumerate() {
        routes[i] = route_edge(rec, &blocks, &lnodes, &layer_h, &track_h, &cfg, width);
        for p in &routes[i].points {
            width = width.max(p.x + cfg.padding);
            height = height.max(p.y + cfg.padding);
        }
    }

    let edge_kinds = recs.iter().map(|r| r.kind).collect();
    LayoutResult { positions, node_sizes, routes, width, height, node_layers, edge_kinds }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: find loops (iterative port of layout.ts findLoops)
// ═══════════════════════════════════════════════════════════════════════════

fn find_loops(blocks: &mut [Block], root: usize) {
    // (block, loop_ids_by_depth)
    let mut work: Vec<(usize, Vec<usize>)> = vec![(root, vec![root])];
    while let Some((b, mut ids)) = work.pop() {
        if blocks[b].loop_id >= 0 {
            continue;
        }
        if blocks[b].is_loop_header {
            let parent = *ids.last().expect("ids never empty");
            // A header acting as a (fallback) root would otherwise become its
            // own parent and hang the loop-height walk. Upstream never hits
            // this because headers always have a backedge pred (never roots).
            blocks[b].parent_loop = if parent == b { None } else { Some(parent) };
            ids.push(b);
        }
        if blocks[b].loop_depth > ids.len() as i32 - 1 {
            blocks[b].loop_depth = ids.len() as i32 - 1;
        }
        if blocks[b].loop_depth < ids.len() as i32 - 1 {
            ids.truncate(blocks[b].loop_depth as usize + 1);
        }
        blocks[b].loop_id = ids[blocks[b].loop_depth as usize] as i64;

        if !blocks[b].is_backedge {
            // Reverse push so the first successor is processed first.
            for k in (0..blocks[b].succs.len()).rev() {
                let s = blocks[b].succs[k];
                work.push((s, ids.clone()));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: layer assignment (iterative port of layout.ts assignLayers)
// ═══════════════════════════════════════════════════════════════════════════

fn assign_layers(blocks: &mut [Block], root: usize) {
    assign_layers_from(blocks, root, 0);
}

fn assign_layers_from(blocks: &mut [Block], root: usize, start_layer: i64) {
    enum Frame {
        /// Visit `block` at `layer`; `defer_check` carries the parent's loop
        /// depth and loop id so the loop-exit deferral test runs exactly when
        /// the recursive caller would have run it.
        Enter { block: usize, layer: i64, defer_check: Option<(i32, i64)> },
        /// Process the next deferred loop exit of `block` (live list).
        FinishHeader { block: usize, layer: i64, next: usize },
    }

    let mut work = vec![Frame::Enter { block: root, layer: start_layer, defer_check: None }];
    while let Some(frame) = work.pop() {
        match frame {
            Frame::Enter { block: b, layer, defer_check } => {
                if let Some((parent_depth, parent_loop_id)) = defer_check {
                    if blocks[b].loop_depth < parent_depth {
                        // Outgoing edge from the parent's loop — defer to the
                        // parent's loop header.
                        if parent_loop_id >= 0 {
                            let header = parent_loop_id as usize;
                            blocks[header].outgoing_edges.push(b);
                            continue;
                        }
                    }
                }
                if blocks[b].is_backedge {
                    let succ = blocks[b].succs[0];
                    blocks[b].layer = blocks[succ].layer;
                    continue;
                }
                if layer <= blocks[b].layer {
                    continue;
                }
                blocks[b].layer = blocks[b].layer.max(layer);

                // Update loop heights up the nesting chain (capped in case
                // degraded metadata ever produces a parent cycle).
                let mut header = if blocks[b].loop_id >= 0 { Some(blocks[b].loop_id as usize) } else { None };
                let mut steps = 0;
                while let Some(h) = header {
                    blocks[h].loop_height = blocks[h].loop_height.max(blocks[b].layer - blocks[h].layer + 1);
                    header = blocks[h].parent_loop.filter(|&p| p != h);
                    steps += 1;
                    if steps > blocks.len() {
                        break;
                    }
                }

                if blocks[b].is_loop_header {
                    work.push(Frame::FinishHeader { block: b, layer, next: 0 });
                }
                for k in (0..blocks[b].succs.len()).rev() {
                    let s = blocks[b].succs[k];
                    work.push(Frame::Enter {
                        block: s,
                        layer: layer + 1,
                        defer_check: Some((blocks[b].loop_depth, blocks[b].loop_id)),
                    });
                }
            }
            Frame::FinishHeader { block: b, layer, next } => {
                // Live iteration: outgoing_edges can grow while we process it.
                if next < blocks[b].outgoing_edges.len() {
                    let succ = blocks[b].outgoing_edges[next];
                    work.push(Frame::FinishHeader { block: b, layer, next: next + 1 });
                    work.push(Frame::Enter { block: succ, layer: layer + blocks[b].loop_height, defer_check: None });
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 4: layout graph construction (port of layout.ts makeLayoutNodes)
// ═══════════════════════════════════════════════════════════════════════════

fn connect(lnodes: &mut [LNode], from: usize, from_port: usize, to: usize) {
    lnodes[from].dst[from_port] = to;
    if !lnodes[to].src.contains(&from) {
        lnodes[to].src.push(from);
    }
}

fn make_layout_nodes(
    blocks: &mut [Block],
    layer_index: &HashMap<i64, usize>,
    layer_count: usize,
    cfg: &Config,
) -> (Vec<LNode>, Vec<Vec<usize>>) {
    let mut blocks_by_layer: Vec<Vec<usize>> = vec![Vec::new(); layer_count];
    for b in 0..blocks.len() {
        blocks_by_layer[layer_index[&blocks[b].layer]].push(b);
    }

    #[derive(Clone)]
    struct IncompleteEdge {
        src: usize,
        src_port: usize,
        dst_block: usize,
    }

    let mut lnodes: Vec<LNode> = Vec::new();
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); layer_count];
    let mut active: Vec<IncompleteEdge> = Vec::new();
    let mut latest_backedge_dummy: HashMap<usize, usize> = HashMap::new();

    // Every loop's channel (its backedge dummy chain) must span from the
    // header's layer down to its DEEPEST feeder, so the channel column stays
    // reserved on every layer the ascent crosses. For well-formed input the
    // loop's own members already cover this range; inferred overlapping
    // loops can have feeders outside the stack-walk membership.
    let mut chain_range: Vec<(usize, usize, usize)> = Vec::new(); // (header, top li, bottom li)
    for h in 0..blocks.len() {
        let Some(back) = blocks[h].backedge else { continue };
        let top = layer_index[&blocks[h].layer];
        let mut bottom = top;
        for &f in &blocks[back].preds {
            bottom = bottom.max(layer_index[&blocks[f].layer]);
        }
        chain_range.push((h, top, bottom));
    }

    let new_lnode = |lnodes: &mut Vec<LNode>, block: Option<usize>, dst_block: Option<usize>, w: f64, h: f64, slots: usize, layer: usize, ports: usize, cfg: &Config| -> usize {
        lnodes.push(LNode {
            x: cfg.padding,
            y: cfg.padding,
            w,
            h,
            block,
            dst_block,
            dst: vec![NONE; slots],
            src: Vec::new(),
            joints: Vec::new(),
            flags: 0,
            layer,
            ports,
        });
        lnodes.len() - 1
    };

    for li in 0..layer_count {
        let here = &blocks_by_layer[li];
        let here_set: HashSet<usize> = here.iter().copied().collect();

        // Remove terminating edges (preserving original active order).
        let mut terminating: Vec<IncompleteEdge> = Vec::new();
        active.retain(|e| {
            if here_set.contains(&e.dst_block) {
                terminating.push(e.clone());
                false
            } else {
                true
            }
        });

        // Create dummies for pass-through edges, coalescing by destination.
        let mut dummies_by_dest: HashMap<usize, usize> = HashMap::new();
        for k in 0..active.len() {
            let dstb = active[k].dst_block;
            let dummy = match dummies_by_dest.get(&dstb) {
                Some(&d) => d,
                None => {
                    let d = new_lnode(&mut lnodes, None, Some(dstb), 0.0, 0.0, 1, li, 1, cfg);
                    layers[li].push(d);
                    dummies_by_dest.insert(dstb, d);
                    d
                }
            };
            let (src, port) = (active[k].src, active[k].src_port);
            connect(&mut lnodes, src, port, dummy);
            active[k].src = dummy;
            active[k].src_port = 0;
        }
        // NOTE: upstream keeps duplicate active entries (multiple edges now
        // flowing through one shared dummy) — they're harmless and their
        // positions drive later layers' dummy creation order.

        // Which loops need a backedge dummy on this layer, and after which
        // block (the LAST block of that loop on this layer).
        let mut pending_loop_dummies: Vec<(usize, usize)> = Vec::new(); // (loop header id, block)
        for &b in here {
            let mut current = if blocks[b].loop_id >= 0 { Some(blocks[b].loop_id as usize) } else { None };
            let mut steps = 0;
            while let Some(h) = current {
                if !blocks[h].is_loop_header {
                    break;
                }
                if let Some(entry) = pending_loop_dummies.iter_mut().find(|(lh, _)| *lh == h) {
                    entry.1 = b;
                } else {
                    pending_loop_dummies.push((h, b));
                }
                current = blocks[h].parent_loop.filter(|&p| p != h);
                steps += 1;
                if steps > blocks.len() {
                    break;
                }
            }
            // A backedge block always counts as a member of its own loop:
            // the loop's dummy must be created AFTER the backedge block's
            // node so the imminent dummy has something to connect to. (With
            // inferred overlapping loops the stack-walk can give the
            // backedge block a different loop_id, missing this.)
            if blocks[b].is_backedge {
                if let Some(h) = (0..blocks.len()).find(|&h| blocks[h].backedge == Some(b)) {
                    if let Some(entry) = pending_loop_dummies.iter_mut().find(|(lh, _)| *lh == h) {
                        entry.1 = b;
                    } else {
                        pending_loop_dummies.push((h, b));
                    }
                }
            }
        }
        // Loops whose feeders extend below their member layers still need
        // their channel reserved here (owner: the layer's last block).
        if let Some(&owner) = here.last() {
            for &(h, top, bottom) in &chain_range {
                if li >= top && li <= bottom && !pending_loop_dummies.iter().any(|(lh, _)| *lh == h) {
                    pending_loop_dummies.push((h, owner));
                }
            }
        }
        // And the inverse: degraded loop_ids can request a chain dummy on a
        // layer the loop doesn't span (e.g. above its header). Drop those —
        // a dummy there would never connect to anything.
        pending_loop_dummies.retain(|&(h, _)| {
            chain_range.iter().any(|&(ch, top, bottom)| ch == h && li >= top && li <= bottom)
        });

        // Create real nodes (and their backedge dummies).
        let mut backedge_edges: Vec<IncompleteEdge> = Vec::new();
        for &b in here {
            let slots = blocks[b].succs.len();
            let node = new_lnode(&mut lnodes, Some(b), None, blocks[b].w, blocks[b].h, slots, li, slots.max(1), cfg);
            for e in &terminating {
                if e.dst_block == b {
                    connect(&mut lnodes, e.src, e.src_port, node);
                }
            }
            layers[li].push(node);
            blocks[b].layout_node = node;

            for &(loop_header, _owner) in pending_loop_dummies.iter().filter(|&&(_, owner)| owner == b) {
                let Some(back) = blocks[loop_header].backedge else { continue };
                let dummy = new_lnode(&mut lnodes, None, Some(back), 0.0, 0.0, 1, li, 1, cfg);
                if let Some(&latest) = latest_backedge_dummy.get(&back) {
                    connect(&mut lnodes, dummy, 0, latest);
                } else {
                    lnodes[dummy].flags |= IMMINENT_BACKEDGE_DUMMY;
                    let target = blocks[back].layout_node;
                    if target != NONE {
                        connect(&mut lnodes, dummy, 0, target);
                    }
                }
                layers[li].push(dummy);
                latest_backedge_dummy.insert(back, dummy);
            }

            if blocks[b].is_backedge {
                let succ = blocks[b].succs[0];
                let target = blocks[succ].layout_node;
                if target != NONE {
                    connect(&mut lnodes, node, 0, target);
                }
            } else {
                for (i, &succ) in blocks[b].succs.iter().enumerate() {
                    if blocks[succ].is_backedge {
                        backedge_edges.push(IncompleteEdge { src: node, src_port: i, dst_block: succ });
                    } else {
                        active.push(IncompleteEdge { src: node, src_port: i, dst_block: succ });
                    }
                }
            }
        }

        for e in &backedge_edges {
            if let Some(&dummy) = latest_backedge_dummy.get(&e.dst_block) {
                connect(&mut lnodes, e.src, e.src_port, dummy);
            }
        }
    }

    // Prune orphaned backedge dummy chains.
    let mut removed: HashSet<usize> = HashSet::new();
    for li in 0..layer_count {
        for k in 0..layers[li].len() {
            let n = layers[li][k];
            if lnodes[n].block.is_none() {
                if let Some(dstb) = lnodes[n].dst_block {
                    if blocks[dstb].is_backedge && lnodes[n].src.is_empty() && !removed.contains(&n) {
                        let mut cur = n;
                        loop {
                            if lnodes[cur].block.is_some() || !lnodes[cur].src.is_empty() {
                                break;
                            }
                            // prune
                            for di in 0..lnodes[cur].dst.len() {
                                let d = lnodes[cur].dst[di];
                                if d != NONE {
                                    lnodes[d].src.retain(|&s| s != cur);
                                }
                            }
                            removed.insert(cur);
                            if lnodes[cur].dst.len() != 1 || lnodes[cur].dst[0] == NONE {
                                break;
                            }
                            cur = lnodes[cur].dst[0];
                        }
                    }
                }
            }
        }
    }
    for layer in layers.iter_mut() {
        layer.retain(|n| !removed.contains(n));
    }

    // Mark leftmost / rightmost dummy runs.
    for layer in &layers {
        for &n in layer {
            if lnodes[n].block.is_none() {
                lnodes[n].flags |= LEFTMOST_DUMMY;
            } else {
                break;
            }
        }
        for &n in layer.iter().rev() {
            if lnodes[n].block.is_none() {
                lnodes[n].flags |= RIGHTMOST_DUMMY;
            } else {
                break;
            }
        }
    }

    (lnodes, layers)
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 5: horizontal straightening (port of layout.ts straightenEdges)
// ═══════════════════════════════════════════════════════════════════════════

fn port_step(n: &LNode, cfg: &Config) -> f64 {
    if n.ports <= 1 {
        return cfg.port_step;
    }
    cfg.port_step.min(((n.w - 2.0 * cfg.port_start) / (n.ports - 1) as f64).max(0.0))
}

fn output_x(n: &LNode, port: usize, cfg: &Config) -> f64 {
    n.x + cfg.port_start + port_step(n, cfg) * port as f64
}

fn input_x(n: &LNode, cfg: &Config) -> f64 {
    n.x + cfg.port_start
}

fn push_neighbors(lnodes: &mut [LNode], layer: &[usize], cfg: &Config) {
    for i in 0..layer.len().saturating_sub(1) {
        let n = layer[i];
        let nb = layer[i + 1];
        let first_non_dummy = lnodes[n].block.is_none() && lnodes[nb].block.is_some();
        let right = lnodes[n].x + lnodes[n].w + if first_non_dummy { cfg.port_start } else { 0.0 } + cfg.gap;
        lnodes[nb].x = lnodes[nb].x.max(right);
    }
}

fn straighten_edges(lnodes: &mut Vec<LNode>, layers: &mut [Vec<usize>], blocks: &[Block], cfg: &Config) {
    let push_into_loops = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        for layer in layers {
            for &n in layer {
                let Some(b) = lnodes[n].block else { continue };
                if blocks[b].loop_id >= 0 {
                    let header_node = blocks[blocks[b].loop_id as usize].layout_node;
                    if header_node != NONE {
                        lnodes[n].x = lnodes[n].x.max(lnodes[header_node].x);
                    }
                }
            }
        }
    };

    let straighten_dummy_runs = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        let mut run_x: HashMap<usize, f64> = HashMap::new();
        for layer in layers {
            for &n in layer {
                if lnodes[n].block.is_none() {
                    if let Some(d) = lnodes[n].dst_block {
                        let e = run_x.entry(d).or_insert(0.0);
                        *e = e.max(lnodes[n].x);
                    }
                }
            }
        }
        for layer in layers {
            for &n in layer {
                if lnodes[n].block.is_none() {
                    if let Some(d) = lnodes[n].dst_block {
                        lnodes[n].x = run_x[&d];
                    }
                }
            }
        }
        for layer in layers {
            push_neighbors(lnodes, layer, cfg);
        }
    };

    let suck_in_leftmost_dummies = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        let mut run_x: HashMap<usize, f64> = HashMap::new();
        for layer in layers {
            let mut i = 0;
            let mut next_x = 0.0;
            while i < layer.len() {
                if lnodes[layer[i]].flags & LEFTMOST_DUMMY == 0 {
                    next_x = lnodes[layer[i]].x;
                    break;
                }
                i += 1;
            }
            if i == layer.len() && i > 0 {
                // all dummies; fall back to the last one's position
                next_x = lnodes[*layer.last().expect("non-empty")].x;
            }
            let mut next_x = next_x - cfg.gap - cfg.port_start;
            let mut k = i;
            while k > 0 {
                k -= 1;
                let dummy = layer[k];
                let mut max_safe = next_x;
                for si in 0..lnodes[dummy].src.len() {
                    let s = lnodes[dummy].src[si];
                    let port = lnodes[s].dst.iter().position(|&d| d == dummy).unwrap_or(0);
                    let sx = lnodes[s].x + port as f64 * port_step(&lnodes[s], cfg);
                    if sx < max_safe {
                        max_safe = sx;
                    }
                }
                lnodes[dummy].x = max_safe;
                next_x = lnodes[dummy].x - cfg.gap;
                if let Some(d) = lnodes[dummy].dst_block {
                    let e = run_x.entry(d).or_insert(f64::INFINITY);
                    *e = e.min(max_safe);
                }
            }
        }
        for layer in layers {
            for &n in layer {
                if lnodes[n].block.is_none() && lnodes[n].flags & LEFTMOST_DUMMY != 0 {
                    if let Some(d) = lnodes[n].dst_block {
                        if let Some(&x) = run_x.get(&d) {
                            if x.is_finite() {
                                lnodes[n].x = x;
                            }
                        }
                    }
                }
            }
        }
    };

    let dbg_block: Option<usize> = std::env::var("ION_DEBUG_CHILD").ok().and_then(|v| v.parse().ok());
    let straighten_children = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        for li in 0..layers.len().saturating_sub(1) {
            push_neighbors(lnodes, &layers[li], cfg);
            let mut last_shifted: i64 = -1;
            for &n in &layers[li] {
                for port in 0..lnodes[n].dst.len() {
                    let dst = lnodes[n].dst[port];
                    if dst == NONE {
                        continue;
                    }
                    let dst_index = layers[li + 1].iter().position(|&x| x == dst).map(|i| i as i64).unwrap_or(-1);
                    let dbg_li: Option<usize> = std::env::var("ION_DEBUG_CHILD_LI").ok().and_then(|v| v.parse().ok());
                    let interesting = (dbg_block.is_some() && lnodes[dst].block == dbg_block) || dbg_li == Some(li);
                    if dst_index > last_shifted && lnodes[dst].src.first() == Some(&n) {
                        let target = output_x(&lnodes[n], port, cfg) - cfg.port_start;
                        if interesting {
                            let verdict = if target > lnodes[dst].x { "SHIFT" } else { "no (target<=x)" };
                            eprintln!("CHILD li={} src=ln{}(block={:?},x={:.2}) port={} -> target={:.2} dst.x={:.2} dst_index={} last_shifted={} => {}", li, n, lnodes[n].block, lnodes[n].x, port, target, lnodes[dst].x, dst_index, last_shifted, verdict);
                        }
                        if target > lnodes[dst].x {
                            lnodes[dst].x = target;
                            last_shifted = dst_index;
                        }
                    } else if interesting {
                        eprintln!("CHILD li={} src=ln{}(block={:?}) port={} SKIPPED dst_index={} last_shifted={} src0={:?}", li, n, lnodes[n].block, port, dst_index, last_shifted, lnodes[dst].src.first());
                    }
                }
            }
        }
    };

    let straighten_conservative = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        for layer in layers {
            for i in (0..layer.len()).rev() {
                let n = layer[i];
                // NOTE: replicates an upstream quirk (`if (!node.blockId)`):
                // the block with id 0 is never nudged, exactly like layout.ts.
                let Some(b) = lnodes[n].block else { continue };
                if b == 0 {
                    continue;
                }
                if blocks[b].is_backedge {
                    continue;
                }

                let mut deltas: Vec<f64> = Vec::new();
                for si in 0..lnodes[n].src.len() {
                    let parent = lnodes[n].src[si];
                    let port = lnodes[parent].dst.iter().position(|&d| d == n).unwrap_or(0);
                    deltas.push(output_x(&lnodes[parent], port, cfg) - input_x(&lnodes[n], cfg));
                }
                for port in 0..lnodes[n].dst.len() {
                    let dst = lnodes[n].dst[port];
                    if dst == NONE {
                        continue;
                    }
                    if lnodes[dst].block.is_none() {
                        if let Some(db) = lnodes[dst].dst_block {
                            if blocks[db].is_backedge {
                                continue;
                            }
                        }
                    }
                    deltas.push(input_x(&lnodes[dst], cfg) - output_x(&lnodes[n], port, cfg));
                }

                if deltas.iter().any(|&d| d == 0.0) {
                    continue;
                }
                let mut deltas: Vec<f64> = deltas.into_iter().filter(|&d| d > 0.0).collect();
                deltas.sort_by(|a, b| a.total_cmp(b));

                for d in deltas {
                    let mut overlaps = false;
                    for j in (i + 1)..layer.len() {
                        let other = layer[j];
                        if lnodes[other].flags & RIGHTMOST_DUMMY != 0 {
                            continue;
                        }
                        let a1 = lnodes[n].x + d;
                        let a2 = lnodes[n].x + d + lnodes[n].w;
                        let b1 = lnodes[other].x - cfg.gap;
                        let b2 = lnodes[other].x + lnodes[other].w + cfg.gap;
                        if a2 >= b1 && a1 <= b2 {
                            overlaps = true;
                            break;
                        }
                    }
                    if !overlaps {
                        lnodes[n].x += d;
                        break;
                    }
                }
            }
            push_neighbors(lnodes, layer, cfg);
        }
    };

    let nearly_straight_up = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        for li in (0..layers.len()).rev() {
            push_neighbors(lnodes, &layers[li], cfg);
            for &n in &layers[li] {
                for si in 0..lnodes[n].src.len() {
                    let s = lnodes[n].src[si];
                    if lnodes[s].block.is_some() {
                        continue;
                    }
                    if (lnodes[s].x - lnodes[n].x).abs() <= cfg.nearly_straight {
                        let x = lnodes[s].x.max(lnodes[n].x);
                        lnodes[s].x = x;
                        lnodes[n].x = x;
                    }
                }
            }
        }
    };

    let nearly_straight_down = |lnodes: &mut Vec<LNode>, layers: &[Vec<usize>]| {
        for li in 0..layers.len() {
            push_neighbors(lnodes, &layers[li], cfg);
            for &n in &layers[li] {
                if lnodes[n].dst.is_empty() {
                    continue;
                }
                let dst = lnodes[n].dst[0];
                if dst == NONE || lnodes[dst].block.is_some() {
                    continue;
                }
                if (lnodes[dst].x - lnodes[n].x).abs() <= cfg.nearly_straight {
                    let x = lnodes[dst].x.max(lnodes[n].x);
                    lnodes[dst].x = x;
                    lnodes[n].x = x;
                }
            }
        }
    };

    let dbg = std::env::var("ION_DEBUG_PASSES").is_ok();
    let snap = |lnodes: &Vec<LNode>, label: &str| {
        if dbg {
            let xs: Vec<i64> = lnodes.iter().map(|n| n.x as i64).collect();
            eprintln!("PASS {label}: {xs:?}");
        }
    };
    for _ in 0..cfg.layout_iterations {
        straighten_children(lnodes, layers);
        snap(lnodes, "children");
        push_into_loops(lnodes, layers);
        snap(lnodes, "loops");
        straighten_dummy_runs(lnodes, layers);
        snap(lnodes, "dummyruns");
    }
    straighten_dummy_runs(lnodes, layers);
    snap(lnodes, "dummyruns2");
    for _ in 0..cfg.nearly_straight_iterations {
        nearly_straight_up(lnodes, layers);
        nearly_straight_down(lnodes, layers);
    }
    snap(lnodes, "nearly");
    straighten_conservative(lnodes, layers);
    snap(lnodes, "conservative");
    straighten_dummy_runs(lnodes, layers);
    snap(lnodes, "dummyruns3");
    suck_in_leftmost_dummies(lnodes, layers);
    snap(lnodes, "suckin");
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 6: joints / tracks (port of layout.ts computeJoints)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_joints(lnodes: &mut [LNode], layers: &[Vec<usize>], blocks: &[Block], cfg: &Config) -> Vec<f64> {
    let mut track_h = Vec::with_capacity(layers.len());
    for layer in layers {
        #[derive(Clone, Copy)]
        struct Joint {
            x1: f64,
            x2: f64,
            node: usize,
            port: usize,
            dst: usize,
        }
        let mut joints_list = Vec::<Joint>::new();
        for &n in layer {
            lnodes[n].joints = vec![0.0; lnodes[n].dst.len()];
            if let Some(b) = lnodes[n].block {
                if blocks[b].is_backedge {
                    continue;
                }
            }
            for port in 0..lnodes[n].dst.len() {
                let dst = lnodes[n].dst[port];
                if dst == NONE {
                    continue;
                }
                let x1 = output_x(&lnodes[n], port, cfg);
                let x2 = input_x(&lnodes[dst], cfg);
                if (x2 - x1).abs() < 2.0 * cfg.radius {
                    continue;
                }
                joints_list.push(Joint { x1, x2, node: n, port, dst });
            }
        }
        joints_list.sort_by(|a, b| a.x1.total_cmp(&b.x1));

        let mut rightward: Vec<Vec<Joint>> = Vec::new();
        let mut leftward: Vec<Vec<Joint>> = Vec::new();
        'joints: for joint in joints_list {
            let set = if joint.x2 - joint.x1 >= 0.0 { &mut rightward } else { &mut leftward };
            let mut last_valid: Option<usize> = None;
            for i in (0..set.len()).rev() {
                let mut overlaps = false;
                for other in &set[i] {
                    if other.dst == joint.dst {
                        set[i].push(joint);
                        continue 'joints;
                    }
                    let (al, ar) = (joint.x1.min(joint.x2), joint.x1.max(joint.x2));
                    let (bl, br) = (other.x1.min(other.x2), other.x1.max(other.x2));
                    if ar >= bl && al <= br {
                        overlaps = true;
                        break;
                    }
                }
                if overlaps {
                    break;
                }
                last_valid = Some(i);
            }
            match last_valid {
                Some(i) => set[i].push(joint),
                None => set.push(vec![joint]),
            }
        }

        let h = (rightward.len() + leftward.len()).saturating_sub(1) as f64 * cfg.track_step;
        let mut off = -h / 2.0;
        for track in rightward.iter().rev().chain(leftward.iter()) {
            for joint in track {
                lnodes[joint.node].joints[joint.port] = off;
            }
            off += cfg.track_step;
        }
        track_h.push(h);
    }
    track_h
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 7: vertical placement (port of layout.ts verticalize)
// ═══════════════════════════════════════════════════════════════════════════

fn verticalize(lnodes: &mut [LNode], layers: &[Vec<usize>], track_h: &[f64], cfg: &Config) -> Vec<f64> {
    let mut layer_h = vec![0.0f64; layers.len()];
    let mut y = cfg.padding;
    for (li, layer) in layers.iter().enumerate() {
        let mut h: f64 = 0.0;
        for &n in layer {
            lnodes[n].y = y;
            h = h.max(lnodes[n].h);
        }
        layer_h[li] = h;
        y += h + cfg.track_pad + track_h[li] + cfg.track_pad;
    }
    layer_h
}

// ═══════════════════════════════════════════════════════════════════════════
// Routing: convert the layout graph into one bezier route per input edge
// ═══════════════════════════════════════════════════════════════════════════

fn bend_y(n: &LNode, port: usize, layer_h: &[f64], track_h: &[f64], cfg: &Config) -> f64 {
    n.y + layer_h.get(n.layer).copied().unwrap_or(0.0)
        + cfg.track_pad
        + track_h.get(n.layer).copied().unwrap_or(0.0) / 2.0
        + n.joints.get(port).copied().unwrap_or(0.0)
}

fn route_edge(
    rec: &EdgeRec,
    blocks: &[Block],
    lnodes: &[LNode],
    layer_h: &[f64],
    track_h: &[f64],
    cfg: &Config,
    graph_width: f64,
) -> Route {
    match rec.kind {
        EdgeKind::Invalid => Route::default(),
        EdgeKind::SelfLoop => {
            let ln = blocks[rec.tail].layout_node;
            if ln == NONE {
                return Route::default();
            }
            self_loop_route(&lnodes[ln], lnodes[ln].dst.len(), cfg)
        }
        _ => {
            let tail_node = blocks[rec.tail].layout_node;
            let head_node = blocks[rec.head].layout_node;
            if tail_node == NONE || head_node == NONE {
                return Route::default();
            }
            // Backedge block → its header: the horizontal loop-header arrow.
            if blocks[rec.tail].is_backedge && rec.slot != NONE {
                return loop_header_route(&lnodes[tail_node], &lnodes[head_node], lnodes, cfg);
            }
            if rec.slot == NONE {
                // Edge dropped from the layout (mislabeled feedback):
                // route around the right side of the drawing.
                let tail = &lnodes[tail_node];
                let head = &lnodes[head_node];
                return feedback_channel_route(
                    output_x(tail, 0, cfg),
                    tail.y + tail.h,
                    tail.y + tail.h + 42.0,
                    graph_width + cfg.gap,
                    input_x(head, cfg),
                    head.y,
                    cfg.radius,
                );
            }
            let first = lnodes[tail_node].dst.get(rec.slot).copied().unwrap_or(NONE);
            if first == NONE {
                // Slot never got connected (degraded input); route around the
                // right side instead of dropping the edge.
                let tail = &lnodes[tail_node];
                let head = &lnodes[head_node];
                return feedback_channel_route(
                    output_x(tail, rec.slot, cfg),
                    tail.y + tail.h,
                    tail.y + tail.h + 42.0,
                    graph_width + cfg.gap,
                    input_x(head, cfg),
                    head.y,
                    cfg.radius,
                );
            }
            // Edge into a backedge block goes through the backedge dummy
            // chain (upward along the loop channel).
            let into_backedge = blocks
                .get(rec.head)
                .map(|h| h.is_backedge)
                .unwrap_or(false)
                || matches!(lnodes[first].dst_block, Some(db) if blocks[db].is_backedge);
            if into_backedge {
                backedge_channel_route(tail_node, rec.slot, blocks, lnodes, layer_h, track_h, cfg)
            } else {
                chain_route(tail_node, rec.slot, rec.head, blocks, lnodes, layer_h, track_h, cfg)
            }
        }
    }
}

/// Follow a forward edge's dummy chain down to its destination block.
fn chain_route(
    tail_node: usize,
    slot: usize,
    head_block: usize,
    blocks: &[Block],
    lnodes: &[LNode],
    layer_h: &[f64],
    track_h: &[f64],
    cfg: &Config,
) -> Route {
    let tail = &lnodes[tail_node];
    let x0 = output_x(tail, slot, cfg);
    let y0 = tail.y + tail.h;
    let mut b = Ortho::start(x0, y0);
    let mut cur_x = x0;
    let mut cur = tail_node;
    let mut port = slot;
    let mut guard = 0;
    let head_lnode = blocks[head_block].layout_node;
    loop {
        guard += 1;
        if guard > lnodes.len() + 2 {
            break;
        }
        let next = lnodes[cur].dst.get(port).copied().unwrap_or(NONE);
        if next == NONE {
            break;
        }
        let bend = bend_y(&lnodes[cur], port, layer_h, track_h, cfg);
        let next_x = input_x(&lnodes[next], cfg);
        if (next_x - cur_x).abs() > 1e-9 {
            b.to(cur_x, bend);
            b.to(next_x, bend);
            cur_x = next_x;
        }
        if next == head_lnode {
            break;
        }
        cur = next;
        port = 0;
    }
    let head = &lnodes[head_lnode];
    let end_y = (head.y - ARROW_INSET).max(y0 + 1.0);
    b.to(cur_x, end_y);
    Route { points: b.beziers(cfg.radius), arrow_tip: IonPoint { x: cur_x, y: head.y } }
}

/// A feeder edge into a backedge block: down into this layer's track band,
/// across to the loop channel (the backedge dummy chain), up the chain, and
/// horizontally into the backedge block's right edge — or, for synthesized
/// zero-size backedge blocks, all the way into the loop header itself.
fn backedge_channel_route(
    tail_node: usize,
    slot: usize,
    blocks: &[Block],
    lnodes: &[LNode],
    layer_h: &[f64],
    track_h: &[f64],
    cfg: &Config,
) -> Route {
    let tail = &lnodes[tail_node];
    let x0 = output_x(tail, slot, cfg);
    let y0 = tail.y + tail.h;
    let mut b = Ortho::start(x0, y0);

    let mut cur = lnodes[tail_node].dst.get(slot).copied().unwrap_or(NONE);
    if cur == NONE {
        return Route { points: b.beziers(cfg.radius), arrow_tip: IonPoint { x: x0, y: y0 } };
    }

    // First hop: down to the tail's track band, across to the chain.
    let bend = bend_y(tail, slot, layer_h, track_h, cfg);
    let mut cur_x = input_x(&lnodes[cur], cfg);
    if (cur_x - x0).abs() > 1e-9 {
        b.to(x0, bend);
        b.to(cur_x, bend);
    }

    // Walk the chain upward until the imminent dummy (or a real node).
    let mut guard = 0;
    while lnodes[cur].block.is_none() && lnodes[cur].flags & IMMINENT_BACKEDGE_DUMMY == 0 {
        guard += 1;
        if guard > lnodes.len() + 2 {
            break;
        }
        let next = lnodes[cur].dst.first().copied().unwrap_or(NONE);
        if next == NONE {
            break;
        }
        // Jog at the band just above this dummy's layer if columns differ.
        let next_x = if lnodes[next].block.is_none() { input_x(&lnodes[next], cfg) } else { cur_x };
        if (next_x - cur_x).abs() > 1e-9 {
            let ym = lnodes[cur].y - cfg.track_pad;
            b.to(cur_x, ym);
            b.to(next_x, ym);
            cur_x = next_x;
        }
        cur = next;
    }

    // cur is now the imminent dummy (its dst is the backedge block's node)
    // or, defensively, a real node.
    let (back_node, entry_from_dummy_y) = if lnodes[cur].block.is_some() {
        (cur, lnodes[cur].y + HEADER_PUSHDOWN)
    } else {
        let t = lnodes[cur].dst.first().copied().unwrap_or(NONE);
        (t, lnodes[cur].y + HEADER_PUSHDOWN + cfg.radius)
    };
    if back_node == NONE {
        let end = *b.last();
        return Route { points: b.beziers(cfg.radius), arrow_tip: end };
    }
    let back = &lnodes[back_node];
    let backedge_y = back.y + HEADER_PUSHDOWN;
    b.to(cur_x, entry_from_dummy_y.max(backedge_y));

    if let Some(bb) = back.block {
        if blocks[bb].w == 0.0 {
            // Synthesized backedge block: keep going into the header.
            if let Some(&header) = blocks[bb].succs.first() {
                let hn = blocks[header].layout_node;
                if hn != NONE {
                    return approach_from_right(b, cur_x, hn, lnodes, cfg);
                }
            }
        }
    }
    approach_from_right(b, cur_x, back_node, lnodes, cfg)
}

/// Finish a route by entering `target`'s right edge horizontally at
/// header-arrow height, coming in leftward from `cur_x`. When a sibling node
/// blocks the approach, dodge through the clear band above the layer
/// (upstream draws straight through and relies on z-order).
fn approach_from_right(mut b: Ortho, cur_x: f64, target_node: usize, lnodes: &[LNode], cfg: &Config) -> Route {
    let t = &lnodes[target_node];
    let hy = t.y + HEADER_PUSHDOWN.min(t.h.max(HEADER_PUSHDOWN * 2.0) / 2.0);
    let target_right = t.x + t.w;
    let blocked = lnodes.iter().enumerate().any(|(i, n)| {
        i != target_node
            && n.block.is_some()
            && n.layer == t.layer
            && n.w > 0.0
            && n.x + n.w > target_right + 1.0
            && n.x < cur_x - 1.0
            && hy >= n.y
            && hy <= n.y + n.h
    });
    if blocked {
        let up_y = t.y - cfg.track_pad / 2.0;
        let dodge_x = target_right + cfg.gap / 2.0;
        b.to(cur_x, up_y);
        b.to(dodge_x, up_y);
        b.to(dodge_x, hy);
    } else {
        b.to(cur_x, hy);
    }
    b.to(target_right + ARROW_INSET, hy);
    Route { points: b.beziers(cfg.radius), arrow_tip: IonPoint { x: target_right, y: hy } }
}

/// The horizontal arrow from a backedge block into its loop header. When a
/// sibling node sits in between, dodge through the clear band above the
/// layer (upstream draws straight through and relies on z-order).
fn loop_header_route(back: &LNode, header: &LNode, lnodes: &[LNode], cfg: &Config) -> Route {
    let y = back.y + HEADER_PUSHDOWN;
    let target = header.x + header.w;
    let blocked = lnodes.iter().any(|n| {
        n.block.is_some()
            && n.layer == header.layer
            && n.w > 0.0
            && n.x + n.w > target + 1.0
            && n.x < back.x - 1.0
            && y >= n.y
            && y <= n.y + n.h
    });
    let mut b = Ortho::start(back.x, y);
    if blocked {
        let up_y = header.y - cfg.track_pad / 2.0;
        let dodge_x = target + cfg.gap / 2.0;
        b.to(back.x - cfg.gap / 2.0, y);
        b.to(back.x - cfg.gap / 2.0, up_y);
        b.to(dodge_x, up_y);
        b.to(dodge_x, y);
    }
    b.to(target + ARROW_INSET, y);
    Route { points: b.beziers(cfg.radius), arrow_tip: IonPoint { x: target, y } }
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry: orthogonal path builder
// ═══════════════════════════════════════════════════════════════════════════

/// Orthogonal polyline accumulator. Push corner points (each move must share
/// an axis with the previous point); `beziers()` converts the polyline into
/// cubic bezier control points with rounded corners.
struct Ortho {
    pts: Vec<IonPoint>,
}

impl Ortho {
    fn start(x: f64, y: f64) -> Self {
        Ortho { pts: vec![IonPoint { x, y }] }
    }

    fn last(&self) -> &IonPoint {
        self.pts.last().expect("Ortho always has a start point")
    }

    fn to(&mut self, x: f64, y: f64) {
        let last = *self.last();
        if (last.x - x).abs() < 1e-9 && (last.y - y).abs() < 1e-9 {
            return;
        }
        // Merge collinear continuations so corner rounding sees clean turns.
        if self.pts.len() >= 2 {
            let prev = self.pts[self.pts.len() - 2];
            if seg_dir(prev, last) == seg_dir(last, IonPoint { x, y }) {
                self.pts.pop();
            }
        }
        self.pts.push(IonPoint { x, y });
    }

    /// Cubic bezier control points (3k+1) tracing the polyline with corners
    /// rounded at up to `radius` (clamped to half of each adjacent segment).
    fn beziers(&self, radius: f64) -> Vec<IonPoint> {
        const K: f64 = 0.5522847498307936;
        let pts = &self.pts;
        if pts.len() < 2 {
            return vec![pts[0]; 4];
        }
        let n = pts.len();
        let seg_len = |a: IonPoint, b: IonPoint| (b.x - a.x).abs() + (b.y - a.y).abs();
        let mut radii = vec![0.0f64; n];
        for i in 1..n - 1 {
            let d1 = seg_dir(pts[i - 1], pts[i]);
            let d2 = seg_dir(pts[i], pts[i + 1]);
            let perpendicular = (d1.0 == 0) != (d2.0 == 0);
            if perpendicular {
                radii[i] = radius.min(seg_len(pts[i - 1], pts[i]) / 2.0).min(seg_len(pts[i], pts[i + 1]) / 2.0);
            }
        }
        let mut out = vec![pts[0]];
        let mut cur = pts[0];
        for i in 1..n {
            let d = seg_dir_f(pts[i - 1], pts[i]);
            let r_here = if i < n - 1 { radii[i] } else { 0.0 };
            let a = IonPoint { x: pts[i].x - d.0 * r_here, y: pts[i].y - d.1 * r_here };
            if seg_len(cur, a) > 1e-9 {
                push_line(&mut out, cur, a);
                cur = a;
            }
            if i < n - 1 && r_here > 1e-9 {
                let d2 = seg_dir_f(pts[i], pts[i + 1]);
                let b = IonPoint { x: pts[i].x + d2.0 * r_here, y: pts[i].y + d2.1 * r_here };
                out.push(IonPoint { x: cur.x + d.0 * K * r_here, y: cur.y + d.1 * K * r_here });
                out.push(IonPoint { x: b.x - d2.0 * K * r_here, y: b.y - d2.1 * K * r_here });
                out.push(b);
                cur = b;
            }
        }
        if out.len() == 1 {
            push_line(&mut out, cur, *pts.last().expect("non-empty"));
        }
        out
    }
}

fn seg_dir(a: IonPoint, b: IonPoint) -> (i8, i8) {
    let dx = if (b.x - a.x).abs() < 1e-9 { 0 } else if b.x > a.x { 1 } else { -1 };
    let dy = if (b.y - a.y).abs() < 1e-9 { 0 } else if b.y > a.y { 1 } else { -1 };
    (dx, dy)
}

fn seg_dir_f(a: IonPoint, b: IonPoint) -> (f64, f64) {
    let d = seg_dir(a, b);
    (d.0 as f64, d.1 as f64)
}

fn push_line(out: &mut Vec<IonPoint>, a: IonPoint, b: IonPoint) {
    out.push(IonPoint { x: a.x + (b.x - a.x) / 3.0, y: a.y + (b.y - a.y) / 3.0 });
    out.push(IonPoint { x: a.x + 2.0 * (b.x - a.x) / 3.0, y: a.y + 2.0 * (b.y - a.y) / 3.0 });
    out.push(b);
}

/// Down, across to a vertical channel, up/down the channel, and in
/// horizontally (leftward) to the target.
fn feedback_channel_route(x0: f64, y0: f64, exit_y: f64, channel_x: f64, target_x: f64, target_y: f64, r: f64) -> Route {
    let mut b = Ortho::start(x0, y0);
    b.to(x0, exit_y);
    b.to(channel_x, exit_y);
    b.to(channel_x, target_y);
    b.to(target_x + ARROW_INSET, target_y);
    Route { points: b.beziers(r), arrow_tip: IonPoint { x: target_x, y: target_y } }
}

/// Self-loop: exit the bottom of the node, swing right around the node's
/// right edge, and come back in horizontally at the node's upper right.
fn self_loop_route(n: &LNode, port: usize, cfg: &Config) -> Route {
    let x0 = output_x(n, port.min(n.ports.saturating_sub(1)), cfg).min(n.x + n.w - 4.0).max(n.x + 4.0);
    let y0 = n.y + n.h;
    let entry_y = n.y + HEADER_PUSHDOWN.min(n.h / 2.0);
    let channel_x = (n.x + n.w).max(x0) + 28.0;
    let exit_y = y0 + 14.0;
    feedback_channel_route(x0, y0, exit_y, channel_x, n.x + n.w, entry_y, cfg.radius.min(7.0))
}

fn reaches(start: usize, target: usize, adj: &[Vec<usize>]) -> bool {
    let mut stack = vec![start];
    let mut seen = vec![false; adj.len()];
    while let Some(n) = stack.pop() {
        if n == target {
            return true;
        }
        if seen[n] {
            continue;
        }
        seen[n] = true;
        stack.extend(adj[n].iter().copied());
    }
    false
}
