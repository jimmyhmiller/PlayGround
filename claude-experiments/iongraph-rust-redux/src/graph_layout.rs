// Port of Graph.ts layout algorithms and rendering

use crate::graph::*;
use crate::layout_provider::{LayoutProvider, Vec2};

#[derive(Clone)]
struct Joint {
    x1: f64,
    x2: f64,
    src_node_idx: usize,
    src_port: usize,
    dst_node_idx: usize,
}

impl<P: LayoutProvider> Graph<P> {
    pub fn layout(&mut self) -> (Vec<Vec<LayoutNode>>, Vec<f64>, Vec<f64>) {
        // Find roots (blocks with no predecessors)
        let roots: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.predecessors.is_empty())
            .map(|(idx, _)| idx)
            .collect();

        // Make roots into pseudo loop headers
        for &root_idx in &roots {
            self.loops.push(LoopHeader {
                block_idx: root_idx,
                loop_height: 0.0,
                parent_loop: None,
                outgoing_edges: Vec::new(),
                backedge: 0, // Will error if accessed
            });
        }

        // Run layout algorithms
        for &root_idx in &roots {
            self.find_loops(root_idx, None);
            self.layer(root_idx, 0);
        }

        let mut layout_nodes_by_layer = self.make_layout_nodes();
        self.straighten_edges(&mut layout_nodes_by_layer);
        let track_heights = self.finagle_joints(&mut layout_nodes_by_layer);
        let layer_heights = self.verticalize(&mut layout_nodes_by_layer, &track_heights);

        (layout_nodes_by_layer, layer_heights, track_heights)
    }

    fn find_loops(&mut self, block_idx: usize, loop_ids_by_depth: Option<Vec<String>>) {
        let mut loop_ids = loop_ids_by_depth.unwrap_or_else(|| vec![self.blocks[block_idx].id.clone()]);

        // Early out if we already have a loop ID
        if !self.blocks[block_idx].loop_id.is_empty() {
            return;
        }

        let is_loop_header = self.blocks[block_idx].is_loop_header();

        if is_loop_header {
            // This is a true loop header
            let parent_id = loop_ids[loop_ids.len() - 1].clone();

            // Find or create loop header for this block
            let current_loop_idx = self.loops.iter().position(|lh| lh.block_idx == block_idx);

            let current_loop_idx = if let Some(idx) = current_loop_idx {
                idx
            } else {
                // Create new loop header
                let backedge_idx = self.blocks[block_idx]
                    .predecessors
                    .iter()
                    .find(|&&pred_idx| self.blocks[pred_idx].is_backedge())
                    .copied()
                    .unwrap_or(0);

                self.loops.push(LoopHeader {
                    block_idx,
                    loop_height: 0.0,
                    parent_loop: None,
                    outgoing_edges: Vec::new(),
                    backedge: backedge_idx,
                });
                self.loops.len() - 1
            };

            // Set parent loop relationship
            if let Some(&parent_idx) = self.blocks_by_id.get(&parent_id) {
                let parent_loop_idx = self.loops.iter().position(|lh| lh.block_idx == parent_idx);
                self.loops[current_loop_idx].parent_loop = parent_loop_idx;
            }

            loop_ids.push(self.blocks[block_idx].id.clone());
        }

        // Adjust loop depth if necessary
        let loop_depth = self.blocks[block_idx].loop_depth as usize;
        if loop_depth < loop_ids.len() - 1 {
            loop_ids.truncate(loop_depth + 1);
        } else if loop_depth >= loop_ids.len() {
            // Force block back to lesser loop depth
            self.blocks[block_idx].loop_depth = (loop_ids.len() - 1) as u32;
        }

        self.blocks[block_idx].loop_id = loop_ids[self.blocks[block_idx].loop_depth as usize].clone();

        // Recurse to successors (except for backedges)
        if !self.blocks[block_idx].is_backedge() {
            let succs = self.blocks[block_idx].succs.clone();
            for succ_idx in succs {
                self.find_loops(succ_idx, Some(loop_ids.clone()));
            }
        }
    }

    fn layer(&mut self, block_idx: usize, layer: i32) {
        if self.blocks[block_idx].is_backedge() {
            let succ_layer = self.blocks[self.blocks[block_idx].succs[0]].layer;
            self.blocks[block_idx].layer = succ_layer;
            return;
        }

        if layer <= self.blocks[block_idx].layer {
            return;
        }

        self.blocks[block_idx].layer = self.blocks[block_idx].layer.max(layer);
        self.num_layers = self
            .num_layers
            .max((self.blocks[block_idx].layer + 1) as usize);

        // Update loop heights for all parent loops
        // TypeScript recalculates height for each parent loop level based on that loop's own layer
        let block_loop_id = self.blocks[block_idx].loop_id.clone();
        let block_layer = self.blocks[block_idx].layer;

        if let Some(&loop_header_idx) = self.blocks_by_id.get(&block_loop_id) {
            // Update this loop and all parent loops
            let mut current_loop_opt = self
                .loops
                .iter()
                .position(|lh| lh.block_idx == loop_header_idx);

            while let Some(current_loop_idx) = current_loop_opt {
                // Calculate height relative to THIS loop header's layer
                let loop_header_layer = self.blocks[self.loops[current_loop_idx].block_idx].layer;
                let height = block_layer - loop_header_layer + 1;
                self.loops[current_loop_idx].loop_height =
                    self.loops[current_loop_idx].loop_height.max(height as f64);
                current_loop_opt = self.loops[current_loop_idx].parent_loop;
            }
        }

        // Track outgoing edges and layer successors
        let block_loop_depth = self.blocks[block_idx].loop_depth;
        let succs = self.blocks[block_idx].succs.clone();
        let block_loop_id = self.blocks[block_idx].loop_id.clone();

        for succ_idx in succs {
            let succ_loop_depth = self.blocks[succ_idx].loop_depth;
            if succ_loop_depth < block_loop_depth {
                // This is an outgoing edge from the current loop
                // Track it on the loop header to be layered later
                if let Some(&loop_header_idx) = self.blocks_by_id.get(&block_loop_id) {
                    if let Some(loop_idx) = self
                        .loops
                        .iter()
                        .position(|lh| lh.block_idx == loop_header_idx)
                    {
                        if !self.loops[loop_idx].outgoing_edges.contains(&succ_idx) {
                            self.loops[loop_idx].outgoing_edges.push(succ_idx);
                        }
                    }
                }
            } else {
                self.layer(succ_idx, layer + 1);
            }
        }

        // If this block is a true loop header, layer its outgoing edges
        if self.blocks[block_idx].is_loop_header() {
            if let Some(loop_idx) = self.loops.iter().position(|lh| lh.block_idx == block_idx) {
                let outgoing_edges = self.loops[loop_idx].outgoing_edges.clone();
                let loop_height = self.loops[loop_idx].loop_height as i32;

                for succ_idx in outgoing_edges {
                    self.layer(succ_idx, layer + loop_height);
                }
            }
        }
    }

    fn make_layout_nodes(&mut self) -> Vec<Vec<LayoutNode>> {
        // Helper to connect layout nodes - uses global indices
        fn connect_nodes(
            layout_nodes: &mut [Vec<LayoutNode>],
            from_layer: usize,
            from_idx: usize,
            from_port: usize,
            to_layer: usize,
            to_idx: usize,
        ) {
            // Calculate global index for destination
            let global_to = layout_nodes.iter().take(to_layer).map(|layer| layer.len()).sum::<usize>() + to_idx;

            // Calculate global index for source
            let global_from = layout_nodes.iter().take(from_layer).map(|layer| layer.len()).sum::<usize>() + from_idx;

            // Set destination on source node
            match &mut layout_nodes[from_layer][from_idx] {
                LayoutNode::BlockNode(node) => {
                    while node.dst_nodes.len() <= from_port {
                        node.dst_nodes.push(0);
                    }
                    node.dst_nodes[from_port] = global_to;
                }
                LayoutNode::DummyNode(node) => {
                    while node.dst_nodes.len() <= from_port {
                        node.dst_nodes.push(0);
                    }
                    node.dst_nodes[from_port] = global_to;
                }
            }

            // Add source reference on destination node
            match &mut layout_nodes[to_layer][to_idx] {
                LayoutNode::BlockNode(node) => {
                    if !node.src_nodes.contains(&global_from) {
                        node.src_nodes.push(global_from);
                    }
                }
                LayoutNode::DummyNode(node) => {
                    if !node.src_nodes.contains(&global_from) {
                        node.src_nodes.push(global_from);
                    }
                }
            }
        }

        #[derive(Clone)]
        struct IncompleteEdge {
            src_layer: usize,
            src_node: usize,
            src_port: usize,
            dst_block: usize,
        }

        // Group blocks by layer
        let mut blocks_by_layer: Vec<Vec<usize>> = vec![Vec::new(); self.num_layers];
        for (idx, block) in self.blocks.iter().enumerate() {
            if block.layer >= 0 {
                blocks_by_layer[block.layer as usize].push(idx);
            }
        }

        let mut layout_nodes_by_layer: Vec<Vec<LayoutNode>> = vec![Vec::new(); self.num_layers];
        let mut node_id: LayoutNodeID = 0;
        let mut active_edges: Vec<IncompleteEdge> = Vec::new();
        let mut latest_dummies_for_backedges: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new(); // backedge_idx -> (layer, node_idx)

        for (layer, block_indices) in blocks_by_layer.iter().enumerate() {
            // Find edges terminating at this layer
            let mut terminating_edges: Vec<IncompleteEdge> = Vec::new();
            for &block_idx in block_indices {
                active_edges.retain(|edge| {
                    if edge.dst_block == block_idx {
                        terminating_edges.push(edge.clone());
                        false
                    } else {
                        true
                    }
                });
            }

            // Create dummy nodes for active edges
            let mut dummies_by_dest: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for edge in &active_edges {
                let dummy_idx = if let Some(&existing_idx) = dummies_by_dest.get(&edge.dst_block) {
                    // Reuse existing dummy for this destination
                    existing_idx
                } else {
                    // Create new dummy
                    let new_idx = layout_nodes_by_layer[layer].len();
                    layout_nodes_by_layer[layer].push(LayoutNode::DummyNode(DummyNode {
                        id: node_id,
                        pos: Vec2 {
                            x: CONTENT_PADDING,
                            y: CONTENT_PADDING,
                        },
                        size: Vec2 { x: 0.0, y: 0.0 },
                        src_nodes: Vec::new(),
                        dst_nodes: Vec::new(),
                        dst_block: edge.dst_block,
                        joint_offsets: Vec::new(),
                        flags: 0,
                    }));
                    dummies_by_dest.insert(edge.dst_block, new_idx);
                    node_id += 1;
                    new_idx
                };

                // Connect source to dummy
                connect_nodes(
                    &mut layout_nodes_by_layer,
                    edge.src_layer,
                    edge.src_node,
                    edge.src_port,
                    layer,
                    dummy_idx,
                );
            }

            // Update active edges to point to the dummies
            let mut updated_edges = Vec::new();
            for edge in &active_edges {
                if let Some(&dummy_idx) = dummies_by_dest.get(&edge.dst_block) {
                    updated_edges.push(IncompleteEdge {
                        src_layer: layer,
                        src_node: dummy_idx,
                        src_port: 0,
                        dst_block: edge.dst_block,
                    });
                }
            }
            active_edges = updated_edges;

            // Track which blocks will get backedge dummy nodes
            #[derive(Clone)]
            struct PendingLoopDummy {
                loop_id: String,  // Now String instead of BlockID
                block_idx: usize,
            }
            let mut pending_loop_dummies: Vec<PendingLoopDummy> = Vec::new();

            for &block_idx in block_indices {
                // Walk up the parent loop chain (matches TypeScript lines 560-579)
                let mut current_loop_id = self.blocks[block_idx].loop_id.clone();

                loop {
                    // Find the loop header for current_loop_id
                    if let Some(&header_idx) = self.blocks_by_id.get(&current_loop_id) {
                        // Check if this is a true loop header (has "loopheader" attribute)
                        if self.blocks[header_idx].is_loop_header() {
                            // Look for existing pending dummy for this loop
                            if let Some(existing) = pending_loop_dummies
                                .iter_mut()
                                .find(|d| d.loop_id == current_loop_id)
                            {
                                // Update to rightmost block in this loop
                                existing.block_idx = block_idx;
                            } else {
                                // New loop encountered
                                pending_loop_dummies.push(PendingLoopDummy {
                                    loop_id: current_loop_id.clone(),
                                    block_idx,
                                });
                            }

                            // Find the parent loop and continue walking up
                            if let Some(loop_header) =
                                self.loops.iter().find(|lh| lh.block_idx == header_idx)
                            {
                                if let Some(parent_idx) = loop_header.parent_loop {
                                    // Get the parent loop's block ID and continue
                                    current_loop_id =
                                        self.blocks[self.loops[parent_idx].block_idx].id.clone();
                                    continue;
                                }
                            }
                        }
                    }
                    // No more parent loops or invalid loop header
                    break;
                }
            }

            // Create block nodes and backedge edges tracking
            let mut backedge_edges: Vec<IncompleteEdge> = Vec::new();

            for &block_idx in block_indices {
                let node_idx = layout_nodes_by_layer[layer].len();
                layout_nodes_by_layer[layer].push(LayoutNode::BlockNode(BlockNode {
                    id: node_id,
                    pos: Vec2 {
                        x: CONTENT_PADDING,
                        y: CONTENT_PADDING,
                    },
                    size: self.blocks[block_idx].size,
                    src_nodes: Vec::new(),
                    dst_nodes: Vec::new(),
                    joint_offsets: Vec::new(),
                    flags: 0,
                    block: block_idx,
                }));

                // Connect terminating edges
                for edge in &terminating_edges {
                    if edge.dst_block == block_idx {
                        connect_nodes(
                            &mut layout_nodes_by_layer,
                            edge.src_layer,
                            edge.src_node,
                            edge.src_port,
                            layer,
                            node_idx,
                        );
                    }
                }

                self.blocks[block_idx].layout_node = Some(node_id);
                node_id += 1;

                // Create backedge dummies for this block if needed
                for pending in &pending_loop_dummies {
                    if pending.block_idx == block_idx {
                        // Find the backedge block for this loop
                        if let Some(&loop_header_idx) = self.blocks_by_id.get(&pending.loop_id) {
                            // Find backedge from loop header
                            let backedge_idx = self
                                .loops
                                .iter()
                                .find(|lh| lh.block_idx == loop_header_idx)
                                .map(|lh| lh.backedge)
                                .unwrap_or(0);

                            if backedge_idx < self.blocks.len() {
                                // Create backedge dummy
                                let backedge_dummy_idx = layout_nodes_by_layer[layer].len();
                                let mut flags = 0;

                                // Check if there's already a dummy for this backedge
                                if let Some(&(prev_layer, prev_idx)) =
                                    latest_dummies_for_backedges.get(&backedge_idx)
                                {
                                    // Connect to previous dummy
                                    layout_nodes_by_layer[layer].push(LayoutNode::DummyNode(
                                        DummyNode {
                                            id: node_id,
                                            pos: Vec2 {
                                                x: CONTENT_PADDING,
                                                y: CONTENT_PADDING,
                                            },
                                            size: Vec2 { x: 0.0, y: 0.0 },
                                            src_nodes: Vec::new(),
                                            dst_nodes: Vec::new(),
                                            dst_block: backedge_idx,
                                            joint_offsets: Vec::new(),
                                            flags,
                                        },
                                    ));

                                    connect_nodes(
                                        &mut layout_nodes_by_layer,
                                        layer,
                                        backedge_dummy_idx,
                                        0,
                                        prev_layer,
                                        prev_idx,
                                    );
                                } else {
                                    // First dummy for this backedge - mark as imminent
                                    flags |= IMMINENT_BACKEDGE_DUMMY;

                                    layout_nodes_by_layer[layer].push(LayoutNode::DummyNode(
                                        DummyNode {
                                            id: node_id,
                                            pos: Vec2 {
                                                x: CONTENT_PADDING,
                                                y: CONTENT_PADDING,
                                            },
                                            size: Vec2 { x: 0.0, y: 0.0 },
                                            src_nodes: Vec::new(),
                                            dst_nodes: Vec::new(),
                                            dst_block: backedge_idx,
                                            joint_offsets: Vec::new(),
                                            flags,
                                        },
                                    ));

                                    // Connect directly to backedge's layout node
                                    if let Some(backedge_layout_node) =
                                        self.blocks[backedge_idx].layout_node
                                    {
                                        // Find the backedge's layer and index within that layer
                                        let mut backedge_global = backedge_layout_node;
                                        let mut backedge_layer = 0;
                                        let mut backedge_idx_in_layer = 0;

                                        for (l, nodes) in layout_nodes_by_layer.iter().enumerate() {
                                            if backedge_global < nodes.len() {
                                                backedge_layer = l;
                                                backedge_idx_in_layer = backedge_global;
                                                break;
                                            }
                                            backedge_global -= nodes.len();
                                        }

                                        // Connect the backedge dummy to the backedge block
                                        connect_nodes(
                                            &mut layout_nodes_by_layer,
                                            layer,
                                            backedge_dummy_idx,
                                            0,
                                            backedge_layer,
                                            backedge_idx_in_layer,
                                        );
                                    }
                                }

                                latest_dummies_for_backedges
                                    .insert(backedge_idx, (layer, backedge_dummy_idx));
                                node_id += 1;
                            }
                        }
                    }
                }

                // Handle block edges
                let is_backedge = self.blocks[block_idx].is_backedge();
                let succs = self.blocks[block_idx].succs.clone();

                if is_backedge {
                    // Connect backedge to loop header immediately (TypeScript line 641-642)
                    if !succs.is_empty() {
                        let header_idx = succs[0];
                        if let Some(header_layout_node) = self.blocks[header_idx].layout_node {
                            // Find the header's layer and index
                            let mut header_global = header_layout_node;
                            let mut header_layer = 0;
                            let mut header_idx_in_layer = 0;

                            for (l, nodes) in layout_nodes_by_layer.iter().enumerate() {
                                if header_global < nodes.len() {
                                    header_layer = l;
                                    header_idx_in_layer = header_global;
                                    break;
                                }
                                header_global -= nodes.len();
                            }

                            // Connect backedge block to loop header
                            connect_nodes(
                                &mut layout_nodes_by_layer,
                                layer,
                                node_idx,
                                0,
                                header_layer,
                                header_idx_in_layer,
                            );
                        }
                    }
                } else {
                    // Regular block - add edges
                    for (port, succ_idx) in succs.iter().enumerate() {
                        let succ_is_backedge = self.blocks[*succ_idx].is_backedge();

                        if succ_is_backedge {
                            // Track to connect after all backedge dummies are created
                            backedge_edges.push(IncompleteEdge {
                                src_layer: layer,
                                src_node: node_idx,
                                src_port: port,
                                dst_block: *succ_idx,
                            });
                        } else {
                            active_edges.push(IncompleteEdge {
                                src_layer: layer,
                                src_node: node_idx,
                                src_port: port,
                                dst_block: *succ_idx,
                            });
                        }
                    }
                }
            }

            // Connect backedge edges to their dummies
            for edge in backedge_edges {
                if let Some(&(dummy_layer, dummy_idx)) =
                    latest_dummies_for_backedges.get(&edge.dst_block)
                {
                    connect_nodes(
                        &mut layout_nodes_by_layer,
                        edge.src_layer,
                        edge.src_node,
                        edge.src_port,
                        dummy_layer,
                        dummy_idx,
                    );
                }
            }
        }

        // Prune orphaned backedge dummies
        // These are dummies that have no source nodes, which can happen when a loop
        // doesn't actually branch back
        {
            use std::collections::HashSet;

            // Helper to remove a node from its destinations' source lists
            fn prune_node(layout_nodes: &mut [Vec<LayoutNode>], layer: usize, node_idx: usize) {
                let dst_nodes = match &layout_nodes[layer][node_idx] {
                    LayoutNode::BlockNode(n) => n.dst_nodes.clone(),
                    LayoutNode::DummyNode(n) => n.dst_nodes.clone(),
                };

                // Calculate global index for this node
                let global_self = layout_nodes.iter().take(layer).map(|layer| layer.len()).sum::<usize>() + node_idx;

                // Remove this node from all its destinations' src_nodes
                for &global_dst in &dst_nodes {
                    // Find the destination layer and index
                    let mut remaining = global_dst;
                    for nodes in layout_nodes.iter_mut() {
                        if remaining < nodes.len() {
                            match &mut nodes[remaining] {
                                LayoutNode::BlockNode(n) => {
                                    n.src_nodes.retain(|&src| src != global_self);
                                }
                                LayoutNode::DummyNode(n) => {
                                    n.src_nodes.retain(|&src| src != global_self);
                                }
                            }
                            break;
                        }
                        remaining -= nodes.len();
                    }
                }
            }

            // Find orphan roots - backedge dummies with no sources
            let mut orphan_roots: Vec<(usize, usize)> = Vec::new(); // (layer, node_idx)
            for (layer, nodes) in layout_nodes_by_layer.iter().enumerate() {
                for (node_idx, node) in nodes.iter().enumerate() {
                    if let LayoutNode::DummyNode(dummy) = node {
                        // Check if this is a backedge dummy
                        if self.blocks[dummy.dst_block].is_backedge()
                            && dummy.src_nodes.is_empty() {
                                orphan_roots.push((layer, node_idx));
                            }
                    }
                }
            }

            // Track all nodes to remove
            let mut removed_global_ids: HashSet<usize> = HashSet::new();

            // For each orphan root, follow the chain and mark for removal
            for &(start_layer, start_idx) in &orphan_roots {
                let mut current_layer = start_layer;
                let mut current_idx = start_idx;

                loop {
                    // Calculate global ID
                    let global_id = layout_nodes_by_layer.iter().take(current_layer).map(|layer| layer.len()).sum::<usize>() + current_idx;

                    // Check if this is a dummy with no sources
                    let (is_dummy, src_count, dst_nodes) = match &layout_nodes_by_layer
                        [current_layer][current_idx]
                    {
                        LayoutNode::BlockNode(_) => break, // Stop at block nodes
                        LayoutNode::DummyNode(n) => (true, n.src_nodes.len(), n.dst_nodes.clone()),
                    };

                    if !is_dummy || src_count > 0 {
                        break;
                    }

                    // Mark for removal
                    prune_node(&mut layout_nodes_by_layer, current_layer, current_idx);
                    removed_global_ids.insert(global_id);

                    // Move to next node in chain
                    if dst_nodes.len() != 1 {
                        break;
                    }

                    // Find next node
                    let next_global = dst_nodes[0];
                    let mut remaining = next_global;
                    let mut found = false;
                    for (layer, nodes) in layout_nodes_by_layer.iter().enumerate() {
                        if remaining < nodes.len() {
                            current_layer = layer;
                            current_idx = remaining;
                            found = true;
                            break;
                        }
                        remaining -= nodes.len();
                    }

                    if !found {
                        break;
                    }
                }
            }

            // Remove all marked nodes from layers
            let mut global_offset = 0;
            for layer in layout_nodes_by_layer.iter_mut() {
                let layer_start_offset = global_offset;
                let mut node_idx_in_layer = 0;

                layer.retain(|_node| {
                    let global_id = layer_start_offset + node_idx_in_layer;
                    node_idx_in_layer += 1;
                    !removed_global_ids.contains(&global_id)
                });

                global_offset += node_idx_in_layer;
            }
        }

        // Mark leftmost and rightmost dummies
        for nodes in layout_nodes_by_layer.iter_mut() {
            for node in nodes.iter_mut() {
                if let LayoutNode::DummyNode(dummy) = node {
                    dummy.flags |= LEFTMOST_DUMMY;
                } else {
                    break;
                }
            }
            for i in (0..nodes.len()).rev() {
                if let LayoutNode::DummyNode(dummy) = &mut nodes[i] {
                    dummy.flags |= RIGHTMOST_DUMMY;
                } else {
                    break;
                }
            }
        }

        layout_nodes_by_layer
    }

    fn straighten_edges(&mut self, layout_nodes_by_layer: &mut [Vec<LayoutNode>]) {
        use crate::graph::{
            BLOCK_GAP, LAYOUT_ITERATIONS, NEARLY_STRAIGHT_ITERATIONS, PORT_SPACING, PORT_START,
        };

        // Helper: Push nodes to the right if they are too close together
        let push_neighbors = |nodes: &mut [LayoutNode]| {
            for i in 0..nodes.len().saturating_sub(1) {
                let (node_is_dummy, node_size_x, node_pos_x) = match &nodes[i] {
                    LayoutNode::BlockNode(n) => (false, n.size.x, n.pos.x),
                    LayoutNode::DummyNode(n) => (true, n.size.x, n.pos.x),
                };

                let neighbor_is_block = match &nodes[i + 1] {
                    LayoutNode::BlockNode(_) => true,
                    LayoutNode::DummyNode(_) => false,
                };

                // TypeScript line 734: firstNonDummy = node.block === null && neighbor.block !== null
                // This means: current is dummy AND next is block
                let first_non_dummy = node_is_dummy && neighbor_is_block;
                let padding = if first_non_dummy { PORT_START } else { 0.0 };
                let node_right_plus_padding = node_pos_x + node_size_x + padding + BLOCK_GAP;

                match &mut nodes[i + 1] {
                    LayoutNode::BlockNode(neighbor) => {
                        neighbor.pos.x = neighbor.pos.x.max(node_right_plus_padding);
                    }
                    LayoutNode::DummyNode(neighbor) => {
                        neighbor.pos.x = neighbor.pos.x.max(node_right_plus_padding);
                    }
                }
            }
        };

        // Pre-compute loop header info to avoid borrowing issues
        let mut block_to_loop_header: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            let loop_id = block.loop_id.clone();
            if let Some(&loop_header_idx) = self.blocks_by_id.get(&loop_id) {
                // Only include actual loop headers (TypeScript uses asLH which checks this)
                if self.blocks[loop_header_idx].is_loop_header() {
                    if let Some(loop_header_layout_node) = self.blocks[loop_header_idx].layout_node
                    {
                        block_to_loop_header.insert(idx, loop_header_layout_node);
                    }
                }
            }
        }

        // Helper: Push nodes to the right so they fit inside their loop
        let push_into_loops =
            |layout_nodes: &mut [Vec<LayoutNode>],
             block_to_loop: &std::collections::HashMap<usize, usize>| {
                // First, build a map of layout node IDs to their positions
                let mut layout_id_to_pos: std::collections::HashMap<usize, f64> =
                    std::collections::HashMap::new();
                for layer in layout_nodes.iter() {
                    for node in layer {
                        match node {
                            LayoutNode::BlockNode(n) => {
                                layout_id_to_pos.insert(n.id, n.pos.x);
                            }
                            LayoutNode::DummyNode(n) => {
                                layout_id_to_pos.insert(n.id, n.pos.x);
                            }
                        }
                    }
                }

                // Now apply the positions
                for layer_nodes in layout_nodes.iter_mut() {
                    for node in layer_nodes.iter_mut() {
                        if let LayoutNode::BlockNode(block_node) = node {
                            let block_idx = block_node.block;

                            if let Some(&loop_header_layout_id) = block_to_loop.get(&block_idx) {
                                if let Some(&loop_header_x) =
                                    layout_id_to_pos.get(&loop_header_layout_id)
                                {
                                    // Push this node to be at least as far right as the loop header
                                    block_node.pos.x = block_node.pos.x.max(loop_header_x);
                                }
                            }
                        }
                    }
                }
            };

        // Helper: Straighten dummy runs
        let straighten_dummy_runs = |layout_nodes: &mut [Vec<LayoutNode>]| {
            use std::collections::HashMap;

            // Track max position of dummies by destination block
            let mut dummy_line_positions: HashMap<usize, f64> = HashMap::new();

            for layer_nodes in layout_nodes.iter() {
                for node in layer_nodes {
                    if let LayoutNode::DummyNode(dummy) = node {
                        let desired_x = dummy.pos.x;
                        dummy_line_positions
                            .entry(dummy.dst_block)
                            .and_modify(|x| *x = x.max(desired_x))
                            .or_insert(desired_x);
                    }
                }
            }

            // Apply positions to all dummies
            for layer_nodes in layout_nodes.iter_mut() {
                for node in layer_nodes.iter_mut() {
                    if let LayoutNode::DummyNode(dummy) = node {
                        if let Some(&x) = dummy_line_positions.get(&dummy.dst_block) {
                            dummy.pos.x = x;
                        }
                    }
                }
            }

            // Push neighbors after adjusting
            for nodes in layout_nodes.iter_mut() {
                push_neighbors(nodes);
            }
        };

        // Helper: Suck in leftmost dummies
        let suck_in_leftmost_dummies = |layout_nodes: &mut [Vec<LayoutNode>]| {
            use std::collections::HashMap;

            let mut dummy_run_positions: HashMap<usize, f64> = HashMap::new();

            for layer_idx in 0..layout_nodes.len() {
                // Find leftmost non-dummy node
                let mut first_block_idx = 0;
                let mut next_x = 0.0;
                for (i, node) in layout_nodes[layer_idx].iter().enumerate() {
                    if let LayoutNode::BlockNode(block_node) = node {
                        first_block_idx = i;
                        next_x = block_node.pos.x;
                        break;
                    }
                }

                // Walk backward through leftmost dummies
                next_x -= BLOCK_GAP + PORT_START;
                for i in (0..first_block_idx).rev() {
                    // Get dummy info we need
                    let (src_nodes, _dst_block, dummy_id) =
                        if let LayoutNode::DummyNode(dummy) = &layout_nodes[layer_idx][i] {
                            if (dummy.flags & LEFTMOST_DUMMY) == 0 {
                                break;
                            }
                            (dummy.src_nodes.clone(), dummy.dst_block, dummy.id)
                        } else {
                            continue;
                        };

                    let mut max_safe_x = next_x;

                    // Don't let dummies go to the right of their source nodes (TypeScript lines 794-799)
                    if layer_idx > 0 {
                        for &src_id in &src_nodes {
                            // Find source node in previous layer
                            for src_node in &layout_nodes[layer_idx - 1] {
                                let (src_x, src_dst_nodes) = match src_node {
                                    LayoutNode::BlockNode(n) => (n.pos.x, &n.dst_nodes),
                                    LayoutNode::DummyNode(n) => (n.pos.x, &n.dst_nodes),
                                };

                                let src_node_id = match src_node {
                                    LayoutNode::BlockNode(n) => n.id,
                                    LayoutNode::DummyNode(n) => n.id,
                                };

                                if src_node_id == src_id {
                                    // Calculate source port position (TypeScript line 795)
                                    // NOTE: TypeScript does NOT add PORT_START here, only PORT_SPACING!
                                    // TypeScript: srcX = src.pos.x + src.dstNodes.indexOf(dummy) * PORT_SPACING
                                    if let Some(port_idx) =
                                        src_dst_nodes.iter().position(|&id| id == dummy_id)
                                    {
                                        let src_port_x = src_x + (port_idx as f64) * PORT_SPACING;
                                        if src_port_x < max_safe_x {
                                            max_safe_x = src_port_x;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    // Update dummy position
                    if let LayoutNode::DummyNode(dummy) = &mut layout_nodes[layer_idx][i] {
                        dummy.pos.x = max_safe_x;
                        // TypeScript line 801: nextX = dummy.pos.x - BLOCK_GAP (no PORT_START!)
                        next_x = dummy.pos.x - BLOCK_GAP;

                        dummy_run_positions
                            .entry(dummy.dst_block)
                            .and_modify(|x| *x = x.min(max_safe_x))
                            .or_insert(max_safe_x);
                    }
                }
            }

            // Apply min positions to all dummies in a run
            for nodes in layout_nodes.iter_mut() {
                for node in nodes.iter_mut() {
                    if let LayoutNode::DummyNode(dummy) = node {
                        if (dummy.flags & LEFTMOST_DUMMY) != 0 {
                            if let Some(&x) = dummy_run_positions.get(&dummy.dst_block) {
                                dummy.pos.x = x;
                            }
                        }
                    }
                }
            }
        };

        // Helper: Walk down layers, pulling children to align with parents
        let straighten_children = |layout_nodes: &mut [Vec<LayoutNode>]| {
            for layer in 0..layout_nodes.len().saturating_sub(1) {
                push_neighbors(&mut layout_nodes[layer]);

                let mut last_shifted = -1_isize;

                // We need to iterate over nodes and their destinations
                // This requires careful handling to avoid borrow checker issues
                let mut updates: Vec<(usize, f64)> = Vec::new();

                for node in layout_nodes[layer].iter() {
                    let (node_id, node_pos_x, dst_nodes) = match node {
                        LayoutNode::BlockNode(n) => (n.id, n.pos.x, &n.dst_nodes),
                        LayoutNode::DummyNode(n) => (n.id, n.pos.x, &n.dst_nodes),
                    };

                    for (src_port, &dst_global_idx) in dst_nodes.iter().enumerate() {
                        // Skip if dst is a backedge dummy (TypeScript presumably doesn't connect these as dst_nodes)
                        // We need to check the destination node's flags
                        let mut dst_is_backedge_dummy = false;
                        let mut global_count = 0;
                        for dst_layer_nodes in layout_nodes.iter() {
                            for dst_node in dst_layer_nodes.iter() {
                                if global_count == dst_global_idx {
                                    if let LayoutNode::DummyNode(d) = dst_node {
                                        // Check if this is a backedge dummy (IMMINENT_BACKEDGE_DUMMY flag)
                                        use crate::graph::IMMINENT_BACKEDGE_DUMMY;
                                        if (d.flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
                                            dst_is_backedge_dummy = true;
                                        }
                                    }
                                    break;
                                }
                                global_count += 1;
                            }
                            if dst_is_backedge_dummy {
                                break;
                            }
                        }

                        if dst_is_backedge_dummy {
                            continue;
                        }
                        // Convert global index to layer-local index
                        let mut current_count = 0;
                        let mut dst_layer = 0;
                        let mut dst_idx_in_layer = 0;

                        for (l, layer_nodes) in layout_nodes.iter().enumerate() {
                            if dst_global_idx < current_count + layer_nodes.len() {
                                dst_layer = l;
                                dst_idx_in_layer = dst_global_idx - current_count;
                                break;
                            }
                            current_count += layer_nodes.len();
                        }

                        if dst_layer != layer + 1 {
                            continue;
                        }

                        if dst_idx_in_layer as isize > last_shifted {
                            // Check if this node is the first parent (TypeScript line 833)
                            let dst_src_nodes = match &layout_nodes[dst_layer][dst_idx_in_layer] {
                                LayoutNode::BlockNode(n) => &n.src_nodes,
                                LayoutNode::DummyNode(n) => &n.src_nodes,
                            };
                            let is_first_parent = dst_src_nodes.first() == Some(&node_id);

                            if is_first_parent {
                                let src_port_offset = PORT_START + PORT_SPACING * src_port as f64;
                                let dst_port_offset = PORT_START;

                                let dst_pos_x = match &layout_nodes[dst_layer][dst_idx_in_layer] {
                                    LayoutNode::BlockNode(n) => n.pos.x,
                                    LayoutNode::DummyNode(n) => n.pos.x,
                                };

                                let new_x =
                                    dst_pos_x.max(node_pos_x + src_port_offset - dst_port_offset);
                                if new_x != dst_pos_x {
                                    updates.push((dst_idx_in_layer, new_x));
                                    last_shifted = dst_idx_in_layer as isize;
                                }
                            }
                        }
                    }
                }

                // Apply updates
                for (idx, new_x) in updates {
                    match &mut layout_nodes[layer + 1][idx] {
                        LayoutNode::BlockNode(n) => n.pos.x = new_x,
                        LayoutNode::DummyNode(n) => n.pos.x = new_x,
                    }
                }
            }
        };

        // Helper: Straighten nearly straight edges going up
        let straighten_nearly_straight_edges_up = |layout_nodes: &mut [Vec<LayoutNode>]| {
            use crate::graph::NEARLY_STRAIGHT;

            for layer in (0..layout_nodes.len()).rev() {
                push_neighbors(&mut layout_nodes[layer]);

                // Collect adjustments to make (layer_idx, node_idx, new_x)
                let mut adjustments: Vec<(usize, usize, f64)> = Vec::new();

                for (node_idx, node) in layout_nodes[layer].iter().enumerate() {
                    let (node_pos_x, src_nodes) = match node {
                        LayoutNode::BlockNode(n) => (n.pos.x, &n.src_nodes),
                        LayoutNode::DummyNode(n) => (n.pos.x, &n.src_nodes),
                    };

                    for &src_global_idx in src_nodes {
                        // Find the source node
                        let mut src_pos_x = None;
                        let mut src_layer = 0;
                        let mut src_idx_in_layer = 0;
                        let mut current_count = 0;

                        for (l, layer_nodes) in layout_nodes.iter().enumerate() {
                            if src_global_idx < current_count + layer_nodes.len() {
                                src_layer = l;
                                src_idx_in_layer = src_global_idx - current_count;
                                src_pos_x = Some(match &layer_nodes[src_idx_in_layer] {
                                    LayoutNode::BlockNode(n) => (n.pos.x, n.block),
                                    LayoutNode::DummyNode(n) => (n.pos.x, usize::MAX), // Use MAX to indicate dummy
                                });
                                break;
                            }
                            current_count += layer_nodes.len();
                        }

                        if let Some((src_x, block_idx)) = src_pos_x {
                            // Only do this to dummies (block_idx == MAX means dummy)
                            if block_idx != usize::MAX {
                                continue;
                            }

                            let wiggle = (src_x - node_pos_x).abs();
                            if wiggle <= NEARLY_STRAIGHT {
                                // TypeScript lines 929-930: update BOTH nodes to the max position
                                let max_x = src_x.max(node_pos_x);
                                adjustments.push((src_layer, src_idx_in_layer, max_x)); // Update source
                                adjustments.push((layer, node_idx, max_x)); // Update destination
                            }
                        }
                    }
                }

                // Apply adjustments
                for (layer_idx, node_idx, new_x) in adjustments {
                    match &mut layout_nodes[layer_idx][node_idx] {
                        LayoutNode::BlockNode(n) => n.pos.x = new_x,
                        LayoutNode::DummyNode(n) => n.pos.x = new_x,
                    }
                }
            }
        };

        // Helper: Straighten nearly straight edges going down
        let straighten_nearly_straight_edges_down = |layout_nodes: &mut [Vec<LayoutNode>]| {
            use crate::graph::NEARLY_STRAIGHT;

            for layer in 0..layout_nodes.len() {
                push_neighbors(&mut layout_nodes[layer]);

                // Collect adjustments (layer_idx, node_idx, new_x)
                let mut adjustments: Vec<(usize, usize, f64)> = Vec::new();

                for (node_idx, node) in layout_nodes[layer].iter().enumerate() {
                    let (node_pos_x, dst_nodes) = match node {
                        LayoutNode::BlockNode(n) => (n.pos.x, &n.dst_nodes),
                        LayoutNode::DummyNode(n) => (n.pos.x, &n.dst_nodes),
                    };

                    if dst_nodes.is_empty() {
                        continue;
                    }

                    // Only process first destination
                    let dst_global_idx = dst_nodes[0];

                    // Find the destination node
                    let mut dst_pos_x = None;
                    let mut dst_layer = 0;
                    let mut dst_idx_in_layer = 0;
                    let mut current_count = 0;

                    for (l, layer_nodes) in layout_nodes.iter().enumerate() {
                        if dst_global_idx < current_count + layer_nodes.len() {
                            dst_layer = l;
                            dst_idx_in_layer = dst_global_idx - current_count;
                            dst_pos_x = Some(match &layer_nodes[dst_idx_in_layer] {
                                LayoutNode::BlockNode(n) => (n.pos.x, n.block),
                                LayoutNode::DummyNode(n) => (n.pos.x, usize::MAX),
                            });
                            break;
                        }
                        current_count += layer_nodes.len();
                    }

                    if let Some((dst_x, block_idx)) = dst_pos_x {
                        // Only do this to dummies
                        if block_idx != usize::MAX {
                            continue;
                        }

                        let wiggle = (dst_x - node_pos_x).abs();
                        if wiggle <= NEARLY_STRAIGHT {
                            // TypeScript lines 956-957: update BOTH nodes to the max position
                            let max_x = dst_x.max(node_pos_x);
                            adjustments.push((dst_layer, dst_idx_in_layer, max_x)); // Update destination
                            adjustments.push((layer, node_idx, max_x)); // Update source
                        }
                    }
                }

                // Apply adjustments
                for (layer_idx, node_idx, new_x) in adjustments {
                    match &mut layout_nodes[layer_idx][node_idx] {
                        LayoutNode::BlockNode(n) => n.pos.x = new_x,
                        LayoutNode::DummyNode(n) => n.pos.x = new_x,
                    }
                }
            }
        };

        // Pre-compute which blocks are backedges to avoid borrowing issues
        let mut is_backedge_block: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            if block.is_backedge() {
                is_backedge_block.insert(idx);
            }
        }

        let mut is_backedge_dst: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            if block.is_backedge() {
                is_backedge_dst.insert(idx);
            }
        }

        // Helper: Conservative straightening without causing overlaps
        #[allow(clippy::type_complexity)]
        let straighten_conservative =
            |layout_nodes: &mut [Vec<LayoutNode>],
             backedge_blocks: &std::collections::HashSet<usize>,
             backedge_dsts: &std::collections::HashSet<usize>| {
                // Pre-compute all global node info to avoid borrowing issues
                let mut global_node_info: Vec<(
                    usize,
                    f64,
                    Vec<usize>,
                    Vec<usize>,
                    usize,
                    Option<usize>,
                )> = Vec::new(); // (global_idx, pos_x, src_nodes, dst_nodes, flags, dst_block)
                let mut current_global = 0;
                for layer in layout_nodes.iter() {
                    for node in layer {
                        let (pos_x, src_nodes, dst_nodes, flags, dst_block) = match node {
                            LayoutNode::BlockNode(n) => {
                                (n.pos.x, n.src_nodes.clone(), n.dst_nodes.clone(), 0, None)
                            }
                            LayoutNode::DummyNode(n) => (
                                n.pos.x,
                                n.src_nodes.clone(),
                                n.dst_nodes.clone(),
                                n.flags as usize,
                                Some(n.dst_block),
                            ),
                        };
                        global_node_info.push((
                            current_global,
                            pos_x,
                            src_nodes,
                            dst_nodes,
                            flags,
                            dst_block,
                        ));
                        current_global += 1;
                    }
                }

                // Pre-compute layer offsets
                let mut layer_offsets: Vec<usize> = Vec::new();
                let mut offset = 0;
                for layer in layout_nodes.iter() {
                    layer_offsets.push(offset);
                    offset += layer.len();
                }

                // Process each layer
                for (layer_idx, nodes) in layout_nodes.iter_mut().enumerate() {
                    let layer_global_offset = layer_offsets[layer_idx];

                    // Walk right to left
                    for i in (0..nodes.len()).rev() {
                        let global_idx = layer_global_offset + i;

                        // Only do this to block nodes, not backedges
                        let (is_block, is_backedge) = match &nodes[i] {
                            LayoutNode::BlockNode(n) => {
                                let backedge = backedge_blocks.contains(&n.block);
                                (true, backedge)
                            }
                            LayoutNode::DummyNode(_) => (false, false),
                        };

                        if !is_block || is_backedge {
                            continue;
                        }

                        // Get this node's info from pre-computed data
                        let (_, node_pos, src_nodes, dst_nodes, _, _) =
                            &global_node_info[global_idx];
                        let mut deltas_to_try: Vec<f64> = Vec::new();

                        // Check parent nodes
                        for &parent_global in src_nodes {
                            if let Some((_, parent_pos, _, parent_dsts, _, _)) =
                                global_node_info.get(parent_global)
                            {
                                if let Some(port_in_parent) =
                                    parent_dsts.iter().position(|&idx| idx == global_idx)
                                {
                                    let src_port_offset =
                                        PORT_START + port_in_parent as f64 * PORT_SPACING;
                                    let dst_port_offset = PORT_START;
                                    let delta = (parent_pos + src_port_offset)
                                        - (node_pos + dst_port_offset);
                                    deltas_to_try.push(delta);
                                }
                            }
                        }

                        // Check child nodes
                        for (src_port, &dst_global) in dst_nodes.iter().enumerate() {
                            if let Some((_, dst_pos, _, _, _dst_flags, dst_block_opt)) =
                                global_node_info.get(dst_global)
                            {
                                // Skip backedge dummies (TypeScript lines 882-884)
                                // Check if this is a dummy (dst_block_opt.is_some()) AND it leads to a backedge block
                                if let Some(dst_block) = dst_block_opt {
                                    if backedge_dsts.contains(dst_block) {
                                        continue;
                                    }
                                }

                                let src_port_offset = PORT_START + src_port as f64 * PORT_SPACING;
                                let dst_port_offset = PORT_START;
                                let delta =
                                    (dst_pos + dst_port_offset) - (node_pos + src_port_offset);
                                deltas_to_try.push(delta);
                            }
                        }

                        // Filter and sort deltas
                        if deltas_to_try.contains(&0.0) {
                            continue;
                        }

                        deltas_to_try.retain(|&d| d > 0.0);
                        deltas_to_try.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        // Try each delta
                        let (node_pos, node_size) = match &nodes[i] {
                            LayoutNode::BlockNode(n) => (n.pos.x, n.size.x),
                            LayoutNode::DummyNode(n) => (n.pos.x, n.size.x),
                        };

                        for delta in deltas_to_try {
                            let mut overlaps_any = false;
                            for node in nodes.iter().skip(i + 1) {
                                // Ignore rightmost dummies
                                let is_rightmost = match node {
                                    LayoutNode::DummyNode(n) => (n.flags & RIGHTMOST_DUMMY) != 0,
                                    _ => false,
                                };

                                if is_rightmost {
                                    continue;
                                }

                                let (other_pos, other_size) = match node {
                                    LayoutNode::BlockNode(n) => (n.pos.x, n.size.x),
                                    LayoutNode::DummyNode(n) => (n.pos.x, n.size.x),
                                };

                                let a1 = node_pos + delta;
                                let a2 = node_pos + delta + node_size;
                                let b1 = other_pos - BLOCK_GAP;
                                let b2 = other_pos + other_size + BLOCK_GAP;

                                if a2 >= b1 && a1 <= b2 {
                                    overlaps_any = true;
                                    break;
                                }
                            }

                            if !overlaps_any {
                                // Apply delta
                                match &mut nodes[i] {
                                    LayoutNode::BlockNode(n) => n.pos.x += delta,
                                    LayoutNode::DummyNode(n) => n.pos.x += delta,
                                }
                                break;
                            }
                        }
                    }

                    push_neighbors(nodes);
                }
            };

        // Run the passes in order (mimicking TypeScript)
        for _iter in 0..LAYOUT_ITERATIONS {
            straighten_children(layout_nodes_by_layer);
            push_into_loops(layout_nodes_by_layer, &block_to_loop_header);
            straighten_dummy_runs(layout_nodes_by_layer);
        }

        straighten_dummy_runs(layout_nodes_by_layer);

        for _iter in 0..NEARLY_STRAIGHT_ITERATIONS {
            straighten_nearly_straight_edges_up(layout_nodes_by_layer);
            straighten_nearly_straight_edges_down(layout_nodes_by_layer);
        }

        straighten_conservative(layout_nodes_by_layer, &is_backedge_block, &is_backedge_dst);
        straighten_dummy_runs(layout_nodes_by_layer);
        suck_in_leftmost_dummies(layout_nodes_by_layer);
    }

    fn finagle_joints(&mut self, layout_nodes_by_layer: &mut [Vec<LayoutNode>]) -> Vec<f64> {
        use crate::graph::{ARROW_RADIUS, JOINT_SPACING, PORT_SPACING, PORT_START};

        // Build a global map of node positions by global INDEX (not ID)
        let mut global_node_positions: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        let mut global_idx = 0;
        for layer in layout_nodes_by_layer.iter() {
            for node in layer {
                let pos_x = match node {
                    LayoutNode::BlockNode(n) => n.pos.x,
                    LayoutNode::DummyNode(n) => n.pos.x,
                };
                global_node_positions.insert(global_idx, pos_x);
                global_idx += 1;
            }
        }

        let mut track_heights = Vec::new();

        for nodes in layout_nodes_by_layer.iter_mut() {
            // First pass: collect node data
            let mut node_data: Vec<(usize, f64, Vec<usize>, bool)> = Vec::new(); // (id, pos.x, dst_nodes, is_backedge)

            for node in nodes.iter() {
                match node {
                    LayoutNode::BlockNode(block_node) => {
                        let is_backedge = self.blocks[block_node.block].is_backedge();
                        node_data.push((
                            block_node.id,
                            block_node.pos.x,
                            block_node.dst_nodes.clone(),
                            is_backedge,
                        ));
                    }
                    LayoutNode::DummyNode(dummy_node) => {
                        node_data.push((
                            dummy_node.id,
                            dummy_node.pos.x,
                            dummy_node.dst_nodes.clone(),
                            false,
                        ));
                    }
                }
            }

            // Collect all joints
            let mut joints: Vec<Joint> = Vec::new();

            for (node_idx, (_node_id, pos_x, dst_nodes, is_backedge)) in node_data.iter().enumerate()
            {
                if *is_backedge {
                    continue;
                }

                for (src_port, &dst_global_idx) in dst_nodes.iter().enumerate() {
                    let x1 = pos_x + PORT_START + PORT_SPACING * src_port as f64;

                    // Look up the dst position using the global map
                    let dst_pos_x = match global_node_positions.get(&dst_global_idx) {
                        Some(&x) => x,
                        None => {
                            continue; // Skip if dst not found
                        }
                    };
                    let x2 = dst_pos_x + PORT_START;

                    if (x2 - x1).abs() < 2.0 * ARROW_RADIUS {
                        continue;
                    }
                    joints.push(Joint {
                        x1,
                        x2,
                        src_node_idx: node_idx,
                        src_port,
                        dst_node_idx: dst_global_idx,
                    });
                }
            }

            // Initialize joint offsets
            for node in nodes.iter_mut() {
                match node {
                    LayoutNode::BlockNode(n) => {
                        n.joint_offsets = vec![0.0; n.dst_nodes.len()];
                    }
                    LayoutNode::DummyNode(n) => {
                        n.joint_offsets = vec![0.0; n.dst_nodes.len()];
                    }
                }
            }

            // Sort joints by x1
            joints.sort_by(|a, b| a.x1.partial_cmp(&b.x1).unwrap());

            // Greedily assign to tracks
            let mut rightward_tracks: Vec<Vec<Joint>> = Vec::new();
            let mut leftward_tracks: Vec<Vec<Joint>> = Vec::new();

            'next_joint: for joint in joints {
                let track_set = if joint.x2 - joint.x1 >= 0.0 {
                    &mut rightward_tracks
                } else {
                    &mut leftward_tracks
                };

                let mut last_valid_track: Option<usize> = None;

                for (i, track) in track_set.iter().enumerate().rev() {
                    let mut overlaps_with_any = false;

                    for other_joint in track {
                        if joint.dst_node_idx == other_joint.dst_node_idx {
                            // Merge arrows to same destination
                            track_set[i].push(joint);
                            continue 'next_joint;
                        }

                        let al = joint.x1.min(joint.x2);
                        let ar = joint.x1.max(joint.x2);
                        let bl = other_joint.x1.min(other_joint.x2);
                        let br = other_joint.x1.max(other_joint.x2);
                        let overlaps = ar >= bl && al <= br;

                        if overlaps {
                            overlaps_with_any = true;
                            break;
                        }
                    }

                    if overlaps_with_any {
                        break;
                    } else {
                        last_valid_track = Some(i);
                    }
                }

                if let Some(track_idx) = last_valid_track {
                    track_set[track_idx].push(joint);
                } else {
                    track_set.push(vec![joint]);
                }
            }

            // Apply joint offsets
            let num_rightward = rightward_tracks.len();
            let num_leftward = leftward_tracks.len();
            let tracks_height =
                ((num_rightward + num_leftward).saturating_sub(1)) as f64 * JOINT_SPACING;

            let mut track_offset = -tracks_height / 2.0;

            let mut all_tracks = rightward_tracks;
            all_tracks.reverse();
            all_tracks.extend(leftward_tracks);

            for track in all_tracks {
                for joint in track {
                    match &mut nodes[joint.src_node_idx] {
                        LayoutNode::BlockNode(n) => {
                            if joint.src_port < n.joint_offsets.len() {
                                n.joint_offsets[joint.src_port] = track_offset;
                            }
                        }
                        LayoutNode::DummyNode(n) => {
                            if joint.src_port < n.joint_offsets.len() {
                                n.joint_offsets[joint.src_port] = track_offset;
                            }
                        }
                    }
                }
                track_offset += JOINT_SPACING;
            }

            track_heights.push(tracks_height);
        }

        track_heights
    }

    fn verticalize(
        &mut self,
        layout_nodes_by_layer: &mut [Vec<LayoutNode>],
        track_heights: &[f64],
    ) -> Vec<f64> {
        let mut layer_heights: Vec<f64> = vec![0.0; layout_nodes_by_layer.len()];

        let mut next_layer_y = CONTENT_PADDING;
        for (i, nodes) in layout_nodes_by_layer.iter_mut().enumerate() {
            let mut layer_height: f64 = 0.0;

            for node in nodes.iter_mut() {
                match node {
                    LayoutNode::BlockNode(n) => {
                        n.pos.y = next_layer_y;
                        layer_height = layer_height.max(n.size.y);
                    }
                    LayoutNode::DummyNode(n) => {
                        n.pos.y = next_layer_y;
                        layer_height = layer_height.max(n.size.y);
                    }
                }
            }

            layer_heights[i] = layer_height;
            next_layer_y += layer_height + TRACK_PADDING + track_heights[i] + TRACK_PADDING;
        }

        layer_heights
    }
}
