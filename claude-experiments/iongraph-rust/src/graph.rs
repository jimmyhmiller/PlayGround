use crate::types::*;
use std::collections::{HashMap, BTreeMap};

pub struct Graph {
    pub pass: Pass,
    pub blocks: Vec<Block>,
    pub blocks_in_order: Vec<Block>,
    pub blocks_by_num: HashMap<BlockNumber, Block>,
    pub blocks_by_id: HashMap<BlockID, Block>,
    pub loops: Vec<Loop>,
    pub viewport_size: Vec2,
    pub translation: Vec2,
    pub zoom: f64,
    pub target_translation: Vec2,
    pub target_zoom: f64,
    pub animating: bool,
}

impl Graph {
    pub fn new(viewport_size: Vec2, pass: Pass) -> Self {
        let mut graph = Self {
            viewport_size,
            translation: Vec2::new(0.0, 0.0),
            zoom: 1.0,
            target_translation: Vec2::new(0.0, 0.0),
            target_zoom: 1.0,
            animating: false,
            blocks: vec![],
            blocks_in_order: vec![],
            blocks_by_num: HashMap::new(),
            blocks_by_id: HashMap::new(),
            loops: vec![],
            pass,
        };

        graph.initialize_blocks();
        graph.detect_loops();
        
        graph
    }

    fn initialize_blocks(&mut self) {
        // Create unified blocks from MIR and LIR data
        let mut blocks_by_number = BTreeMap::new();

        // Process MIR blocks
        for mir_block in &self.pass.mir.blocks {
            let block = Block {
                id: mir_block.id.clone(),
                number: mir_block.number.clone(),
                mir_block: Some(mir_block.clone()),
                lir_block: None,
                predecessors: mir_block.predecessors.clone(),
                successors: mir_block.successors.clone(),
                loop_depth: mir_block.loop_depth,
                loop_num: 0, // Will be calculated during loop detection
                attributes: mir_block.attributes.clone(),
                layer: 0, // Will be assigned later
                size: Vec2::new(100.0, 50.0), // Default size
                
                // New fields for TypeScript Block IR compatibility
                has_layout_node: true,
                instruction_count: mir_block.instructions.len() as u32,
                lir_instruction_count: 0, // Will be set when LIR data is available
                is_branch: mir_block.successors.len() > 1,
                is_entry: mir_block.predecessors.is_empty(),
                is_exit: mir_block.successors.is_empty(),
                is_merge: mir_block.predecessors.len() > 1,
            };
            blocks_by_number.insert(mir_block.number.clone(), block);
        }

        // Add LIR data to existing blocks or create new ones
        for lir_block in &self.pass.lir.blocks {
            if let Some(block) = blocks_by_number.get_mut(&lir_block.number) {
                block.lir_block = Some(lir_block.clone());
                block.lir_instruction_count = lir_block.instructions.len() as u32;
            } else {
                let block = Block {
                    id: lir_block.id.clone(),
                    number: lir_block.number.clone(),
                    mir_block: None,
                    lir_block: Some(lir_block.clone()),
                    predecessors: vec![],
                    successors: vec![],
                    loop_depth: 0,
                    loop_num: 0,
                    attributes: vec![],
                    layer: 0, // Will be assigned later
                    size: Vec2::new(100.0, 50.0), // Default size
                    
                    // New fields for TypeScript Block IR compatibility
                    has_layout_node: true,
                    instruction_count: 0, // No MIR data available
                    lir_instruction_count: lir_block.instructions.len() as u32,
                    is_branch: false, // Unknown without predecessor/successor info
                    is_entry: false, // Unknown without predecessor/successor info
                    is_exit: false, // Unknown without predecessor/successor info
                    is_merge: false, // Unknown without predecessor/successor info
                };
                blocks_by_number.insert(lir_block.number.clone(), block);
            }
        }

        // Convert to vectors and maps
        self.blocks = blocks_by_number.values().cloned().collect();
        self.blocks_in_order = self.blocks.clone();
        
        // Sort blocks_in_order by block number
        self.blocks_in_order.sort_by_key(|b| b.number.clone());

        // Create lookup maps
        for block in &self.blocks {
            self.blocks_by_num.insert(block.number.clone(), block.clone());
            self.blocks_by_id.insert(block.id.clone(), block.clone());
        }

        // Assign layers using topological sort
        self.assign_layers();
    }

    fn assign_layers(&mut self) {
        // Simple layer assignment using topological ordering
        let mut in_degree: HashMap<BlockNumber, usize> = HashMap::new();
        let mut layers: HashMap<BlockNumber, usize> = HashMap::new();

        // Initialize in-degrees
        for block in &self.blocks {
            in_degree.insert(block.number, block.predecessors.len());
        }

        let mut queue = Vec::new();
        
        // Start with blocks that have no predecessors (entry points)
        for block in &self.blocks {
            if block.predecessors.is_empty() {
                queue.push(block.number);
                layers.insert(block.number, 0);
            }
        }

        // Process blocks in topological order
        while let Some(current) = queue.pop() {
            let current_layer = layers[&current];
            
            if let Some(block) = self.blocks_by_num.get(&current) {
                for &successor in &block.successors {
                    // Skip backedges to avoid cycles
                    if let Some(succ_block) = self.blocks_by_num.get(&successor) {
                        if succ_block.attributes.contains(&"backedge".to_string()) {
                            continue;
                        }
                    }

                    let new_layer = current_layer + 1;
                    let existing_layer = layers.get(&successor).copied().unwrap_or(0);
                    layers.insert(successor, new_layer.max(existing_layer));

                    if let Some(count) = in_degree.get_mut(&successor) {
                        *count -= 1;
                        if *count == 0 {
                            queue.push(successor);
                        }
                    }
                }
            }
        }

        // Handle backedge blocks specially - they get the same layer as their target
        for block in &mut self.blocks {
            if block.attributes.contains(&"backedge".to_string()) {
                if let Some(&successor) = block.successors.first() {
                    if let Some(&target_layer) = layers.get(&successor) {
                        block.layer = target_layer;
                        continue;
                    }
                }
            }
            
            if let Some(&layer) = layers.get(&block.number) {
                block.layer = layer;
            }
        }
        
        // Update lookup maps
        self.blocks_by_num.clear();
        self.blocks_by_id.clear();
        for block in &self.blocks {
            self.blocks_by_num.insert(block.number, block.clone());
            self.blocks_by_id.insert(block.id, block.clone());
        }
    }

    fn detect_loops(&mut self) {
        // Simplified loop detection - only use explicit loop headers and backedges
        let mut loops = Vec::new();

        // Find blocks explicitly marked as loop headers
        for block in &self.blocks {
            if block.attributes.contains(&"loopheader".to_string()) {
                let mut backedges = Vec::new();
                
                // Find backedges pointing to this header
                for other_block in &self.blocks {
                    if other_block.attributes.contains(&"backedge".to_string()) 
                        && other_block.successors.contains(&block.number) {
                        backedges.push(other_block.number);
                    }
                }

                loops.push(Loop {
                    header: block.number,
                    backedges,
                    depth: block.loop_depth,
                });
            }
        }

        self.loops = loops;
    }

    // Helper method to check if there's a path from start to end
    fn is_reachable_path(&self, start: &BlockNumber, end: &BlockNumber) -> bool {
        if start == end {
            return true;
        }

        let mut visited = std::collections::HashSet::new();
        let mut to_visit = vec![*start];

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Some(block) = self.blocks_by_num.get(&current) {
                for &successor in &block.successors {
                    if successor == *end {
                        return true;
                    }
                    if !visited.contains(&successor) {
                        to_visit.push(successor);
                    }
                }
            }
        }

        false
    }

    // Layout algorithm phases will be implemented next
    pub fn layout(&mut self) -> (Vec<Vec<LayoutNode>>, Vec<f64>, Vec<f64>) {
        let mut nodes = self.make_layout_nodes();
        self.straighten_edges(&mut nodes);
        let track_heights = self.finangle_joints(&mut nodes);
        let layer_heights = self.verticalize(&mut nodes, &track_heights);
        
        (nodes, layer_heights, track_heights)
    }

    fn make_layout_nodes(&self) -> Vec<Vec<LayoutNode>> {
        // Group blocks by layer
        let mut blocks_by_layer: HashMap<usize, Vec<&Block>> = HashMap::new();
        let max_layer = self.blocks.iter().map(|b| b.layer).max().unwrap_or(0);
        
        for block in &self.blocks {
            blocks_by_layer.entry(block.layer).or_insert_with(Vec::new).push(block);
        }

        // Initialize layout nodes by layer
        let mut layout_nodes_by_layer: Vec<Vec<LayoutNode>> = vec![Vec::new(); max_layer + 1];
        let mut node_id = 0_u32;
        let mut node_lookup: HashMap<LayoutNodeID, usize> = HashMap::new(); // node_id -> layer
        
        // Track active edges that need dummy nodes
        struct IncompleteEdge {
            src_id: LayoutNodeID,
            src_port: usize,
            dst_block_number: BlockNumber,
        }
        let mut active_edges: Vec<IncompleteEdge> = Vec::new();

        // Process each layer
        for layer in 0..=max_layer {
            let empty_vec = vec![];
            let blocks_in_layer = blocks_by_layer.get(&layer).unwrap_or(&empty_vec);

            // Remove edges that terminate at this layer
            let mut terminating_edges: Vec<IncompleteEdge> = Vec::new();
            active_edges.retain(|edge| {
                if blocks_in_layer.iter().any(|b| b.number == edge.dst_block_number) {
                    terminating_edges.push(IncompleteEdge {
                        src_id: edge.src_id,
                        src_port: edge.src_port,
                        dst_block_number: edge.dst_block_number,
                    });
                    false
                } else {
                    true
                }
            });

            // Create dummy nodes for active edges (edge coalescence)
            let mut dummies_by_dest: HashMap<BlockNumber, LayoutNodeID> = HashMap::new();
            for edge in &active_edges {
                if let Some(&dummy_id) = dummies_by_dest.get(&edge.dst_block_number) {
                    // Connect to existing dummy
                    if let Some(dummy_layer) = node_lookup.get(&dummy_id) {
                        if let Some(dummy) = layout_nodes_by_layer[*dummy_layer].iter_mut().find(|n| n.id == dummy_id) {
                            dummy.src_nodes.push(edge.src_id);
                            // Update source node's dst_nodes
                            for src_layer_idx in 0..*dummy_layer {
                                if let Some(src_node) = layout_nodes_by_layer[src_layer_idx].iter_mut()
                                    .find(|n| n.id == edge.src_id) {
                                    src_node.dst_nodes.push(dummy_id);
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // Create new dummy node
                    let dummy_block = self.blocks_by_num.get(&edge.dst_block_number).unwrap().clone();
                    let mut dummy = LayoutNode::new_dummy_node(node_id, dummy_block, layer);
                    dummy.src_nodes.push(edge.src_id);
                    
                    dummies_by_dest.insert(edge.dst_block_number, node_id);
                    node_lookup.insert(node_id, layer);
                    
                    // Update source node's dst_nodes
                    for src_layer_idx in 0..layer {
                        if let Some(src_node) = layout_nodes_by_layer[src_layer_idx].iter_mut()
                            .find(|n| n.id == edge.src_id) {
                            src_node.dst_nodes.push(node_id);
                            break;
                        }
                    }
                    
                    layout_nodes_by_layer[layer].push(dummy);
                    node_id += 1;
                }
            }

            // Update active edges to point to dummies
            for edge in &mut active_edges {
                if let Some(&dummy_id) = dummies_by_dest.get(&edge.dst_block_number) {
                    // Update the dummy node to have this as a destination connection
                    if let Some(dummy) = layout_nodes_by_layer[layer].iter_mut().find(|n| n.id == dummy_id) {
                        // The dummy should eventually connect to the target block
                        // This will be handled in the next layer when the target appears
                    }
                    edge.src_id = dummy_id;
                    edge.src_port = 0;
                }
            }

            // Create real nodes for blocks in this layer
            for block in blocks_in_layer {
                let mut layout_node = LayoutNode::new_block_node(node_id, (*block).clone(), layer);
                
                // Connect terminating edges to this block
                for edge in &terminating_edges {
                    if edge.dst_block_number == block.number {
                        layout_node.src_nodes.push(edge.src_id);
                        // Also update the source node to point to this destination
                        // Find and update the source node's dst_nodes
                        for layer_idx in 0..layer {
                            if let Some(src_node) = layout_nodes_by_layer[layer_idx].iter_mut()
                                .find(|n| n.id == edge.src_id) {
                                src_node.dst_nodes.push(node_id);
                                break;
                            }
                        }
                    }
                }

                // Add outgoing edges to active edges (except backedges)
                for (port, &successor) in block.successors.iter().enumerate() {
                    if let Some(succ_block) = self.blocks_by_num.get(&successor) {
                        if succ_block.attributes.contains(&"backedge".to_string()) {
                            // Handle backedges immediately (connect directly)
                            if let Some(header_block) = self.blocks_by_num.get(&successor) {
                                // Find the header's layout node (it should be in an earlier layer)
                                for earlier_layer in 0..layer {
                                    if let Some(header_node) = layout_nodes_by_layer[earlier_layer].iter_mut()
                                        .find(|n| n.block.as_ref().map_or(false, |b| b.number == header_block.number)) {
                                        layout_node.dst_nodes.push(header_node.id);
                                        header_node.src_nodes.push(layout_node.id);
                                        break;
                                    }
                                }
                            }
                        } else {
                            // Regular edge - add to active edges if it doesn't terminate immediately
                            if succ_block.layer > layer {
                                active_edges.push(IncompleteEdge {
                                    src_id: node_id,
                                    src_port: port,
                                    dst_block_number: successor,
                                });
                            }
                        }
                    }
                }

                node_lookup.insert(node_id, layer);
                layout_nodes_by_layer[layer].push(layout_node);
                node_id += 1;
            }
        }

        // Mark leftmost and rightmost dummies
        for nodes in &mut layout_nodes_by_layer {
            // Mark leftmost dummies
            for node in nodes.iter_mut() {
                if node.is_dummy() {
                    node.flags |= LEFTMOST_DUMMY;
                } else {
                    break;
                }
            }
            
            // Mark rightmost dummies  
            for node in nodes.iter_mut().rev() {
                if node.is_dummy() {
                    node.flags |= RIGHTMOST_DUMMY;
                } else {
                    break;
                }
            }
        }

        layout_nodes_by_layer
    }

    fn straighten_edges(&self, layout_nodes_by_layer: &mut Vec<Vec<LayoutNode>>) {
        // Helper function to push nodes to the right if they are too close together
        let push_neighbors = |nodes: &mut Vec<LayoutNode>| {
            for i in 0..nodes.len().saturating_sub(1) {
                let (left, right) = nodes.split_at_mut(i + 1);
                let node = &left[i];
                let neighbor = &mut right[0];

                let first_non_dummy = node.is_dummy() && !neighbor.is_dummy();
                let node_right_plus_padding = node.pos.x + node.size.x 
                    + if first_non_dummy { PORT_START } else { 0.0 } 
                    + BLOCK_GAP;
                neighbor.pos.x = neighbor.pos.x.max(node_right_plus_padding);
            }
        };

        // Push nodes to the right so they fit inside their loop
        let push_into_loops = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            // Collect loop header positions first
            let mut loop_header_positions: Vec<(u32, f64)> = Vec::new();
            for layer in nodes_by_layer.iter() {
                for node in layer.iter() {
                    if let Some(block) = &node.block {
                        if block.attributes.contains(&"loopheader".to_string()) {
                            loop_header_positions.push((block.loop_depth, node.pos.x));
                        }
                    }
                }
            }

            // Now apply constraints
            for layer in nodes_by_layer.iter_mut() {
                for node in layer.iter_mut() {
                    if let Some(block) = &node.block {
                        // Check if this block is in a loop and needs alignment
                        if block.loop_depth > 0 {
                            for &(header_depth, header_x) in &loop_header_positions {
                                if header_depth < block.loop_depth {
                                    node.pos.x = node.pos.x.max(header_x);
                                }
                            }
                        }
                    }
                }
            }
        };

        // Straighten dummy runs by aligning dummies to the same destination
        let straighten_dummy_runs = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            // Track max position of dummies by destination block
            let mut dummy_line_positions: HashMap<BlockNumber, f64> = HashMap::new();
            
            for layer in nodes_by_layer.iter() {
                for node in layer.iter() {
                    if let Some(dst_block) = &node.dst_block {
                        let desired_x = node.pos.x;
                        dummy_line_positions.insert(
                            dst_block.number,
                            dummy_line_positions.get(&dst_block.number)
                                .copied()
                                .unwrap_or(0.0)
                                .max(desired_x)
                        );
                    }
                }
            }

            // Apply positions to dummies
            for layer in nodes_by_layer.iter_mut() {
                for node in layer.iter_mut() {
                    if let Some(dst_block) = &node.dst_block {
                        if let Some(&x) = dummy_line_positions.get(&dst_block.number) {
                            node.pos.x = x;
                        }
                    }
                }
            }

            // Push neighbors after repositioning
            for layer in nodes_by_layer.iter_mut() {
                push_neighbors(layer);
            }
        };

        // Walk down the layers, pulling children to align with their parents
        let straighten_children = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            for layer_idx in 0..nodes_by_layer.len().saturating_sub(1) {
                push_neighbors(&mut nodes_by_layer[layer_idx]);

                let mut last_shifted = 0;
                
                // Collect position changes to apply them after iteration
                let mut position_changes: Vec<(usize, f64)> = Vec::new();
                
                for (_node_idx, node) in nodes_by_layer[layer_idx].iter().enumerate() {
                    for (src_port, dst_id) in node.dst_nodes.iter().enumerate() {
                        // Find destination node in next layer
                        if let Some((dst_idx, dst_node)) = nodes_by_layer[layer_idx + 1]
                            .iter()
                            .enumerate()
                            .find(|(_, n)| n.id == *dst_id) {
                            
                            if dst_idx > last_shifted {
                                // Check if this is the first parent of the destination
                                if !dst_node.src_nodes.is_empty() && dst_node.src_nodes[0] == node.id {
                                    let src_port_offset = PORT_START + PORT_SPACING * src_port as f64;
                                    let dst_port_offset = PORT_START;

                                    let x_before = dst_node.pos.x;
                                    let new_x = (node.pos.x + src_port_offset - dst_port_offset).max(dst_node.pos.x);
                                    
                                    if new_x != x_before {
                                        position_changes.push((dst_idx, new_x));
                                        last_shifted = dst_idx;
                                    }
                                }
                            }
                        }
                    }
                }

                // Apply position changes
                for (idx, new_x) in position_changes {
                    nodes_by_layer[layer_idx + 1][idx].pos.x = new_x;
                }
            }
        };

        // The main pass sequence (simplified version of the original)
        for _ in 0..LAYOUT_ITERATIONS {
            straighten_children(layout_nodes_by_layer);
            push_into_loops(layout_nodes_by_layer);
            straighten_dummy_runs(layout_nodes_by_layer);
        }
        
        straighten_dummy_runs(layout_nodes_by_layer);

        // Additional passes could be added here for nearly straight edges, etc.
    }

    fn finangle_joints(&self, layout_nodes_by_layer: &mut Vec<Vec<LayoutNode>>) -> Vec<f64> {
        #[derive(Debug, Clone)]
        struct Joint {
            x1: f64,
            x2: f64,
            src_id: LayoutNodeID,
            src_port: usize,
            dst_id: LayoutNodeID,
        }

        let mut track_heights = Vec::new();

        for layer_idx in 0..layout_nodes_by_layer.len() {
            // First pass: collect node positions for lookup
            let mut node_positions: HashMap<LayoutNodeID, f64> = HashMap::new();
            for layer in layout_nodes_by_layer.iter() {
                for node in layer.iter() {
                    node_positions.insert(node.id, node.pos.x);
                }
            }

            // Second pass: build joints and update joint offsets
            let mut joints: Vec<Joint> = Vec::new();
            
            for node in layout_nodes_by_layer[layer_idx].iter_mut() {
                node.joint_offsets = vec![0.0; node.dst_nodes.len()];

                // Skip backedge blocks
                if let Some(block) = &node.block {
                    if block.attributes.contains(&"backedge".to_string()) {
                        continue;
                    }
                }

                for (src_port, dst_id) in node.dst_nodes.iter().enumerate() {
                    let dst_x = node_positions.get(dst_id).copied().unwrap_or(0.0);

                    let x1 = node.pos.x + PORT_START + PORT_SPACING * src_port as f64;
                    let x2 = dst_x + PORT_START;

                    if (x2 - x1).abs() < 2.0 * ARROW_RADIUS {
                        // Ignore edges that are narrow enough not to render with a joint
                        continue;
                    }

                    joints.push(Joint {
                        x1,
                        x2,
                        src_id: node.id,
                        src_port,
                        dst_id: *dst_id,
                    });
                }
            }

            joints.sort_by(|a, b| a.x1.partial_cmp(&b.x1).unwrap());

            // Greedily sort joints into "tracks" based on whether they overlap horizontally
            let mut rightward_tracks: Vec<Vec<Joint>> = Vec::new();
            let mut leftward_tracks: Vec<Vec<Joint>> = Vec::new();

            'next_joint: for joint in joints {
                let track_set = if joint.x2 - joint.x1 >= 0.0 {
                    &mut rightward_tracks
                } else {
                    &mut leftward_tracks
                };

                let mut last_valid_track: Option<usize> = None;

                for (i, track) in track_set.iter().rev().enumerate() {
                    let track_index = track_set.len() - 1 - i;
                    let mut overlaps_with_any_in_track = false;

                    for other_joint in track {
                        if joint.dst_id == other_joint.dst_id {
                            // Assign the joint to this track to merge arrows
                            track_set[track_index].push(joint);
                            continue 'next_joint;
                        }

                        let al = joint.x1.min(joint.x2);
                        let ar = joint.x1.max(joint.x2);
                        let bl = other_joint.x1.min(other_joint.x2);
                        let br = other_joint.x1.max(other_joint.x2);
                        let overlaps = ar >= bl && al <= br;

                        if overlaps {
                            overlaps_with_any_in_track = true;
                            break;
                        }
                    }

                    if overlaps_with_any_in_track {
                        break;
                    } else {
                        last_valid_track = Some(track_index);
                    }
                }

                if let Some(track_index) = last_valid_track {
                    track_set[track_index].push(joint);
                } else {
                    track_set.push(vec![joint]);
                }
            }

            // Use track info to apply joint offsets to nodes for rendering
            let tracks_height = ((rightward_tracks.len() + leftward_tracks.len()).max(1) - 1) as f64 * JOINT_SPACING;
            let mut track_offset = -tracks_height / 2.0;

            rightward_tracks.reverse();
            let all_tracks = rightward_tracks.into_iter().chain(leftward_tracks.into_iter());

            for track in all_tracks {
                for joint in track {
                    // Find the source node and update its joint offset
                    if let Some(src_node) = layout_nodes_by_layer[layer_idx].iter_mut().find(|n| n.id == joint.src_id) {
                        if joint.src_port < src_node.joint_offsets.len() {
                            src_node.joint_offsets[joint.src_port] = track_offset;
                        }
                    }
                }
                track_offset += JOINT_SPACING;
            }

            track_heights.push(tracks_height);
        }

        track_heights
    }

    fn verticalize(&self, layout_nodes_by_layer: &mut Vec<Vec<LayoutNode>>, track_heights: &[f64]) -> Vec<f64> {
        let mut layer_heights = vec![0.0; layout_nodes_by_layer.len()];
        let mut next_layer_y = CONTENT_PADDING;

        for (i, layer) in layout_nodes_by_layer.iter_mut().enumerate() {
            let mut max_node_height: f64 = 0.0;

            // Set Y positions and calculate layer height
            for node in layer.iter_mut() {
                node.pos.y = next_layer_y;
                max_node_height = max_node_height.max(node.size.y);
            }

            // Layer height is the maximum of node height and track height
            let track_height = track_heights.get(i).copied().unwrap_or(0.0);
            let layer_height = max_node_height.max(track_height);
            layer_heights[i] = layer_height;

            next_layer_y += layer_height + TRACK_PADDING + track_height + TRACK_PADDING;
        }

        layer_heights
    }

    // SVG rendering methods
    pub fn render_svg(&mut self) -> String {
        let (nodes_by_layer, layer_heights, _track_heights) = self.layout();
        
        // Calculate total SVG dimensions
        let mut max_x: f64 = 0.0;
        let mut max_y: f64 = 0.0;
        
        for layer in &nodes_by_layer {
            for node in layer {
                max_x = max_x.max(node.pos.x + node.size.x + CONTENT_PADDING);
                max_y = max_y.max(node.pos.y + node.size.y + CONTENT_PADDING);
            }
        }
        
        let width = max_x + CONTENT_PADDING;
        let height = max_y + CONTENT_PADDING;
        
        let mut svg = String::new();
        svg.push_str(&format!(r#"<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"#, 
            width, height, width, height));
        
        // Add styles
        svg.push_str("
<defs>
    <style>
        .block { fill: #f0f8ff; stroke: #4a90e2; stroke-width: 2; rx: 8; }
        .block-text { font-family: 'Monaco', 'Consolas', monospace; font-size: 12px; fill: #333; text-anchor: middle; dominant-baseline: central; }
        .loop-header { fill: #ffe4e1; stroke: #ff6b6b; }
        .backedge { fill: #e8f5e8; stroke: #4caf50; }
        .arrow { stroke: #666; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
        .arrow.backedge { stroke: #4caf50; stroke-width: 2; stroke-dasharray: 5,5; }
        .dummy { fill: #f9f9f9; stroke: #ccc; stroke-width: 1; }
    </style>
    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\" markerUnits=\"strokeWidth\">
        <polygon points=\"0,0 10,3.5 0,7\" fill=\"#666\"/>
    </marker>
</defs>
");
        
        // Render arrows first (so they appear behind blocks)
        self.render_arrows(&mut svg, &nodes_by_layer);
        
        // Render blocks
        self.render_blocks(&mut svg, &nodes_by_layer);
        
        svg.push_str("</svg>");
        svg
    }
    
    fn render_blocks(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        for layer in nodes_by_layer {
            for node in layer {
                if node.is_dummy() {
                    // Render dummy node as small circle
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="3" class="dummy"/>"#,
                        node.pos.x + node.size.x / 2.0,
                        node.pos.y + node.size.y / 2.0
                    ));
                } else if let Some(block) = &node.block {
                    // Determine block style based on attributes
                    let mut class = "block".to_string();
                    if block.attributes.contains(&"loopheader".to_string()) {
                        class.push_str(" loop-header");
                    } else if block.attributes.contains(&"backedge".to_string()) {
                        class.push_str(" backedge");
                    }
                    
                    // Render block rectangle
                    svg.push_str(&format!(
                        r#"<rect x="{}" y="{}" width="{}" height="{}" class="{}"/>"#,
                        node.pos.x, node.pos.y, node.size.x, node.size.y, class
                    ));
                    
                    // Render block text
                    let block_text = format!("B{}", block.number.0);
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" class="block-text">{}</text>"#,
                        node.pos.x + node.size.x / 2.0,
                        node.pos.y + node.size.y / 2.0,
                        block_text
                    ));
                }
            }
        }
    }
    
    fn render_arrows(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        // Build a lookup map for nodes by ID
        let mut node_lookup: std::collections::HashMap<LayoutNodeID, &LayoutNode> = std::collections::HashMap::new();
        for layer in nodes_by_layer {
            for node in layer {
                node_lookup.insert(node.id, node);
            }
        }
        
        // Render arrows for all nodes
        for layer in nodes_by_layer {
            for node in layer {
                // Render each outgoing connection
                for (port_idx, &dst_id) in node.dst_nodes.iter().enumerate() {
                    if let Some(dst_node) = node_lookup.get(&dst_id) {
                        self.render_single_arrow(svg, node, dst_node, port_idx);
                    }
                }
            }
        }
    }
    
    fn render_single_arrow(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode, port_idx: usize) {
        // Calculate arrow start and end points
        let start_x = if src_node.is_dummy() {
            // For dummy nodes, start from center
            src_node.pos.x
        } else {
            // For real blocks, start from port position
            src_node.pos.x + PORT_START + (PORT_SPACING * port_idx as f64).min(src_node.size.x - PORT_START - 10.0)
        };
        
        let start_y = src_node.pos.y + src_node.size.y;
        
        let end_x = if dst_node.is_dummy() {
            // For dummy nodes, end at center
            dst_node.pos.x
        } else {
            // For real blocks, end at left edge + PORT_START
            dst_node.pos.x + PORT_START
        };
        
        let end_y = dst_node.pos.y;
        
        // Get joint offset for this port if available
        let joint_offset = src_node.joint_offsets.get(port_idx).copied().unwrap_or(0.0);
        
        // Determine if this is a backedge (going backwards in layers)
        let is_backedge = src_node.layer > dst_node.layer;
        
        // Determine arrow style class
        let mut arrow_class = "arrow".to_string();
        if is_backedge {
            arrow_class.push_str(" backedge");
        }
        
        // Check if we need a curved or straight arrow
        let horizontal_distance = (end_x - start_x).abs();
        let needs_curve = horizontal_distance > 2.0 * ARROW_RADIUS || joint_offset != 0.0;
        
        if !needs_curve && !is_backedge {
            // Simple straight arrow
            svg.push_str(&format!(
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" class="{}"/>"#,
                start_x, start_y, end_x, end_y, arrow_class
            ));
        } else {
            // Curved arrow with joints
            let mut path = format!("M {:.1} {:.1}", start_x, start_y);
            
            if is_backedge {
                // Backedges curve around to avoid crossing forward edges
                let control_y = start_y + JOINT_SPACING + joint_offset;
                let return_y = end_y - JOINT_SPACING;
                
                // Curve around the right side if going backwards
                let curve_x = start_x.max(end_x) + 30.0 + joint_offset;
                
                path.push_str(&format!(" L {:.1} {:.1}", start_x, control_y));
                path.push_str(&format!(" L {:.1} {:.1}", curve_x, control_y));
                path.push_str(&format!(" L {:.1} {:.1}", curve_x, return_y));
                path.push_str(&format!(" L {:.1} {:.1}", end_x, return_y));
                path.push_str(&format!(" L {:.1} {:.1}", end_x, end_y));
            } else {
                // Normal forward arrow with potential horizontal routing
                let mid_y = start_y + JOINT_SPACING + joint_offset;
                
                path.push_str(&format!(" L {:.1} {:.1}", start_x, mid_y));
                path.push_str(&format!(" L {:.1} {:.1}", end_x, mid_y));
                path.push_str(&format!(" L {:.1} {:.1}", end_x, end_y));
            }
            
            svg.push_str(&format!(
                r#"<path d="{}" class="{}" fill="none"/>"#,
                path, arrow_class
            ));
        }
    }

    // Navigation methods
    pub fn set_selection(&mut self, _block_ids: Vec<BlockID>, _last_selected: Option<BlockID>) {
        // TODO: Implement selection state management
    }

    pub fn navigate(&mut self, _direction: NavigationDirection) {
        // TODO: Implement navigation logic
    }

    // Animation methods
    pub async fn go_to_coordinates(&mut self, pos: Vec2, zoom: f64, animate: bool) -> Result<(), String> {
        if !animate {
            self.translation = Vec2::new(-pos.x * zoom, -pos.y * zoom);
            self.zoom = zoom;
            return Ok(());
        }

        if self.animating {
            return Ok(()); // Don't start another animation
        }

        self.animating = true;
        self.target_translation = Vec2::new(-pos.x * zoom, -pos.y * zoom);
        self.target_zoom = zoom;

        // TODO: Implement actual animation loop with requestAnimationFrame equivalent
        // For now, just set the target values
        self.translation = self.target_translation.clone();
        self.zoom = self.target_zoom;
        self.animating = false;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum NavigationDirection {
    Up,
    Down,
    Left,
    Right,
}

// Snapshot test data structures to match TypeScript original
#[derive(Debug, Clone, serde::Serialize)]
struct LayoutMetrics {
    block_count: usize,
    layer_count: usize,
    graph_size: Vec2,
    viewport_size: Vec2,
    block_sizes: Vec<BlockSize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BlockSize {
    number: u32,
    size: Vec2,
    layer: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ArrowAnalysis {
    total_arrows: usize,
    arrow_types: Vec<ArrowType>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ArrowType {
    index: usize,
    path_count: usize,
    has_arrowhead: bool,
    main_path_data: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PathAnalysis {
    index: usize,
    has_move_to: bool,
    has_line_to: bool,
    has_arc: bool,
    has_curve: bool,
    path_length: usize,
    is_horizontal: bool,
}

fn calculate_graph_size(nodes_by_layer: &[Vec<LayoutNode>]) -> Vec2 {
    let mut max_x: f64 = 0.0;
    let mut max_y: f64 = 0.0;
    
    for layer in nodes_by_layer {
        for node in layer {
            // Include all nodes in size calculation, including dummy nodes
            // This matches TypeScript behavior where dummy nodes contribute to graph bounds
            max_x = max_x.max(node.pos.x + node.size.x);
            max_y = max_y.max(node.pos.y + node.size.y);
        }
    }
    
    // Add content padding
    Vec2::new(max_x + CONTENT_PADDING, max_y + CONTENT_PADDING)
}

fn analyze_svg_arrows(svg: &str) -> ArrowAnalysis {
    // Parse SVG and extract arrow information
    let line_count = svg.matches("<line").count();
    let path_count = svg.matches("<path").count();
    
    let mut arrow_types = Vec::new();
    
    // Analyze line elements (straight arrows)
    for (index, _) in svg.match_indices("<line").enumerate() {
        arrow_types.push(ArrowType {
            index,
            path_count: 1,
            has_arrowhead: true, // All our arrows have arrowheads from CSS
            main_path_data: format!("Line arrow {}", index),
        });
    }
    
    // Analyze path elements (curved arrows)  
    for (index, _) in svg.match_indices("<path").enumerate() {
        let start_idx = index + line_count;
        arrow_types.push(ArrowType {
            index: start_idx,
            path_count: 1,
            has_arrowhead: true,
            main_path_data: format!("Path arrow {}", start_idx),
        });
    }
    
    ArrowAnalysis {
        total_arrows: line_count + path_count,
        arrow_types,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::*;

    #[test]
    fn test_constructor_with_simple_pass() {
        let pass = create_simple_pass();
        let graph = Graph::new(Vec2::new(800.0, 600.0), pass.clone());

        // Verify basic properties
        assert_eq!(graph.pass.name, pass.name);
        assert_eq!(graph.blocks.len(), 2);
        assert_eq!(graph.blocks_in_order.len(), 2);
        assert_eq!(graph.blocks_by_num.len(), 2);
        assert_eq!(graph.blocks_by_id.len(), 2);
    }

    #[test]
    fn test_viewport_properties() {
        let pass = create_simple_pass();
        let graph = Graph::new(Vec2::new(800.0, 600.0), pass);

        assert_eq!(graph.viewport_size.x, 800.0);
        assert_eq!(graph.viewport_size.y, 600.0);
    }

    #[test]
    fn test_block_linking() {
        let pass = create_simple_pass();
        let graph = Graph::new(Vec2::new(800.0, 600.0), pass);

        let block0 = graph.blocks_by_num.get(&BlockNumber(0)).unwrap();
        let block1 = graph.blocks_by_num.get(&BlockNumber(1)).unwrap();

        assert_eq!(block0.successors, vec![BlockNumber(1)]);
        assert_eq!(block1.predecessors, vec![BlockNumber(0)]);
    }

    #[test]
    fn test_svg_rendering_simple() {
        let pass = create_simple_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        let svg = graph.render_svg();
        
        // Should contain basic SVG structure
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
        
        // Should contain our two blocks  
        assert!(svg.contains("B0"));
        assert!(svg.contains("B1"));
    }
    
    #[test]
    fn test_svg_rendering_complex() {
        let pass = create_complex_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        let svg = graph.render_svg();
        
        // Should contain all 5 blocks
        assert!(svg.contains("B0"));
        assert!(svg.contains("B1"));
        assert!(svg.contains("B2"));
        assert!(svg.contains("B3"));
        assert!(svg.contains("B4"));
        
        // Should have arrows
        assert!(svg.contains("<path"));
        assert!(svg.contains("arrow"));
    }
}
