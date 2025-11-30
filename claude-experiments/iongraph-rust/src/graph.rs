use crate::types::*;
use std::collections::{HashMap, BTreeMap};

// HTML escape function for SVG text
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// Font metrics for monospace text (matching PureSVGTextLayoutProvider)
const CHAR_WIDTH: f64 = 7.0;
const CHAR_HEIGHT: f64 = 14.0;
const PADDING: f64 = 8.0;
const HEADER_HEIGHT: f64 = 30.0;

// Calculate block size based on instruction content
fn calculate_block_size(block: &Block) -> Vec2 {
    if let Some(mir_block) = &block.mir_block {
        let mut max_num_width = 0;
        let mut max_opcode_width = 0;
        let mut max_type_width = 0;

        for ins in &mir_block.instructions {
            // Measure instruction number width
            let num_text = format!("{}", ins.id);
            max_num_width = max_num_width.max(num_text.chars().count());

            // Measure opcode width - use RENDERED length (after escape and ← replacement)
            // IMPORTANT: Use .chars().count() not .len() because ← is 3 bytes but 1 character!
            let opcode_rendered = html_escape(&ins.opcode).replace("&lt;-", "←");
            max_opcode_width = max_opcode_width.max(opcode_rendered.chars().count());

            // Measure type width
            if ins.instruction_type != "None" {
                max_type_width = max_type_width.max(ins.instruction_type.chars().count());
            }
        }

        // Also consider header width
        let header_text = if block.attributes.contains(&"loopheader".to_string()) {
            format!("Block {} (loop header)", block.id.0)
        } else if block.attributes.contains(&"backedge".to_string()) {
            format!("Block {} (backedge)", block.id.0)
        } else {
            format!("Block {}", block.id.0)
        };
        let header_width = header_text.chars().count() as f64 * CHAR_WIDTH;

        // Calculate total width: padding + num + gap + opcode + gap + type + padding
        let content_width = PADDING +
            (max_num_width as f64 * CHAR_WIDTH) + 8.0 +
            (max_opcode_width as f64 * CHAR_WIDTH) + 8.0 +
            (max_type_width as f64 * CHAR_WIDTH) + PADDING;

        let width = content_width.max(header_width + PADDING * 2.0).max(150.0);
        let height = HEADER_HEIGHT + (mir_block.instructions.len() as f64 * CHAR_HEIGHT) + PADDING * 2.0;

        Vec2::new(width, height)
    } else if let Some(lir_block) = &block.lir_block {
        // Similar calculation for LIR blocks
        let mut max_opcode_width = 0;

        for ins in &lir_block.instructions {
            max_opcode_width = max_opcode_width.max(ins.opcode.len());
        }

        let header_text = format!("Block {}", block.id.0);
        let header_width = header_text.len() as f64 * CHAR_WIDTH;

        let content_width = PADDING + (max_opcode_width as f64 * CHAR_WIDTH) + PADDING;
        let width = content_width.max(header_width + PADDING * 2.0).max(150.0);
        let height = HEADER_HEIGHT + (lir_block.instructions.len() as f64 * CHAR_HEIGHT) + PADDING * 2.0;

        Vec2::new(width, height)
    } else {
        // Empty block
        Vec2::new(150.0, 60.0)
    }
}

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
            let mut block = Block {
                id: mir_block.id,
                number: BlockNumber(mir_block.id.0),  // Use ID as number
                mir_block: Some(mir_block.clone()),
                lir_block: None,
                predecessors: mir_block.predecessors.iter().map(|id| BlockNumber(id.0)).collect(),
                successors: mir_block.successors.iter().map(|id| BlockNumber(id.0)).collect(),
                loop_depth: mir_block.loop_depth,
                loop_num: 0, // Will be calculated during loop detection
                attributes: mir_block.attributes.clone(),
                layer: 0, // Will be assigned later
                size: Vec2::new(100.0, 50.0), // Temporary, will be recalculated

                // New fields for TypeScript Block IR compatibility
                has_layout_node: true,
                instruction_count: mir_block.instructions.len() as u32,
                lir_instruction_count: 0, // Will be set when LIR data is available
                is_branch: mir_block.successors.len() > 1,
                is_entry: mir_block.predecessors.is_empty(),
                is_exit: mir_block.successors.is_empty(),
                is_merge: mir_block.predecessors.len() > 1,

                // Loop hierarchy fields (initialized later)
                loop_id: BlockID(0),
                parent_loop: None,
                loop_height: 0,
                outgoing_edges: Vec::new(),
            };

            // Calculate proper size based on content
            block.size = calculate_block_size(&block);

            blocks_by_number.insert(BlockNumber(mir_block.id.0), block);
        }

        // Add LIR data to existing blocks or create new ones
        for lir_block in &self.pass.lir.blocks {
            let lir_number = BlockNumber(lir_block.id.0);  // Use ID as number
            if let Some(block) = blocks_by_number.get_mut(&lir_number) {
                block.lir_block = Some(lir_block.clone());
                block.lir_instruction_count = lir_block.instructions.len() as u32;
                // Recalculate size with LIR data
                block.size = calculate_block_size(block);
            } else {
                let mut block = Block {
                    id: lir_block.id,
                    number: lir_number,
                    mir_block: None,
                    lir_block: Some(lir_block.clone()),
                    predecessors: vec![],
                    successors: vec![],
                    loop_depth: 0,
                    loop_num: 0,
                    attributes: vec![],
                    layer: 0, // Will be assigned later
                    size: Vec2::new(100.0, 50.0), // Temporary

                    // New fields for TypeScript Block IR compatibility
                    has_layout_node: true,
                    instruction_count: 0, // No MIR data available
                    lir_instruction_count: lir_block.instructions.len() as u32,
                    is_branch: false, // Unknown without predecessor/successor info
                    is_entry: false, // Unknown without predecessor/successor info
                    is_exit: false, // Unknown without predecessor/successor info
                    is_merge: false, // Unknown without predecessor/successor info

                    // Loop hierarchy fields (initialized later)
                    loop_id: BlockID(0),
                    parent_loop: None,
                    loop_height: 0,
                    outgoing_edges: Vec::new(),
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

    fn find_loops(&mut self, block_num: BlockNumber, loop_ids_by_depth: Vec<BlockID>) {
        // Get the block ID and check if already processed
        let (block_id, already_processed) = {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            let loop_depth = block.loop_depth as usize;

            // Check if this block was already processed with the same loop context
            let already_processed = if loop_depth < loop_ids_by_depth.len() {
                // Check if loop_id matches what we would assign
                block.loop_id == loop_ids_by_depth[loop_depth] && block.loop_id != BlockID(0)
            } else {
                false
            };

            (block.id, already_processed)
        };

        if already_processed {
            return;
        }

        let mut new_loop_ids_by_depth = loop_ids_by_depth.clone();

        // Check if this is a true loop header
        let is_loop_header = {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            block.attributes.contains(&"loopheader".to_string())
        };

        if is_loop_header {
            // Set parent loop from the current stack
            if !new_loop_ids_by_depth.is_empty() {
                let parent_id = new_loop_ids_by_depth[new_loop_ids_by_depth.len() - 1];
                if let Some(block) = self.blocks_by_num.get_mut(&block_num) {
                    block.parent_loop = Some(parent_id);
                }
            }
            // Add this block to the stack
            new_loop_ids_by_depth.push(block_id);
        }

        // Adjust loop_ids_by_depth based on block's loop_depth
        {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            let loop_depth = block.loop_depth as usize;

            if loop_depth < new_loop_ids_by_depth.len().saturating_sub(1) {
                new_loop_ids_by_depth.truncate(loop_depth + 1);
            } else if loop_depth >= new_loop_ids_by_depth.len() {
                // Adjust loop_depth to match stack size (handle corrupted MIR data)
                if let Some(block) = self.blocks_by_num.get_mut(&block_num) {
                    block.loop_depth = (new_loop_ids_by_depth.len() - 1) as u32;
                }
            }
        }

        // Assign loop_id from the stack
        {
            let block = self.blocks_by_num.get_mut(&block_num).unwrap();
            let loop_depth = block.loop_depth as usize;
            if loop_depth < new_loop_ids_by_depth.len() {
                block.loop_id = new_loop_ids_by_depth[loop_depth];
            }
        }

        // Collect successors (need to avoid borrow checker issues)
        let successors: Vec<BlockNumber> = {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            // Don't recurse into backedges
            if block.attributes.contains(&"backedge".to_string()) {
                vec![]
            } else {
                block.successors.clone()
            }
        };

        // Recursively process successors
        for succ in successors {
            self.find_loops(succ, new_loop_ids_by_depth.clone());
        }
    }

    fn assign_layers_longest_path(&mut self, block_num: BlockNumber, layer: usize) {
        // Check if this is a backedge - assign same layer as target
        {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            if block.attributes.contains(&"backedge".to_string()) {
                if let Some(&successor) = block.successors.first() {
                    if let Some(target_block) = self.blocks_by_num.get(&successor) {
                        let target_layer = target_block.layer;
                        if let Some(block_mut) = self.blocks_by_num.get_mut(&block_num) {
                            block_mut.layer = target_layer;
                        }
                    }
                }
                return;
            }
        }

        // Only process if this layer is greater than the block's current layer
        // Use < instead of <= to allow processing layer 0 on first visit
        {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            if layer < block.layer {
                return;
            }
        }

        // Update the block's layer to the maximum computed so far
        {
            let block_mut = self.blocks_by_num.get_mut(&block_num).unwrap();
            block_mut.layer = block_mut.layer.max(layer);
        }

        // Update loop height information for all containing loops
        {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            let mut current_loop_id = Some(block.loop_id);

            while let Some(loop_id) = current_loop_id {
                if let Some(loop_header) = self.blocks_by_id.get(&loop_id) {
                    let header_num = loop_header.number;
                    let header_layer = loop_header.layer;
                    let block_layer = self.blocks_by_num.get(&block_num).unwrap().layer;
                    let new_height = block_layer - header_layer + 1;

                    if let Some(header_mut) = self.blocks_by_num.get_mut(&header_num) {
                        header_mut.loop_height = header_mut.loop_height.max(new_height);
                    }

                    current_loop_id = self.blocks_by_id.get(&loop_id).and_then(|b| b.parent_loop);
                } else {
                    break;
                }
            }
        }

        // Collect successors and determine which are outgoing edges
        let (successors, outgoing, loop_id, loop_depth) = {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            let successors = block.successors.clone();
            let loop_depth = block.loop_depth;
            let loop_id = block.loop_id;

            let mut regular_succs = Vec::new();
            let mut outgoing_succs = Vec::new();

            for &succ_num in &successors {
                if let Some(succ_block) = self.blocks_by_num.get(&succ_num) {
                    if succ_block.loop_depth < loop_depth {
                        // Edge exits the current loop - defer to loop header
                        outgoing_succs.push(succ_num);
                    } else {
                        // Edge stays within loop - process immediately
                        regular_succs.push(succ_num);
                    }
                } else {
                    regular_succs.push(succ_num);
                }
            }

            (regular_succs, outgoing_succs, loop_id, loop_depth)
        };

        // Add outgoing edges to the loop header's list
        if !outgoing.is_empty() {
            if let Some(loop_header) = self.blocks_by_id.get(&loop_id) {
                let header_num = loop_header.number;
                if let Some(header_mut) = self.blocks_by_num.get_mut(&header_num) {
                    for &outgoing_succ in &outgoing {
                        if !header_mut.outgoing_edges.contains(&outgoing_succ) {
                            header_mut.outgoing_edges.push(outgoing_succ);
                        }
                    }
                }
            }
        }

        // Process regular successors with next layer
        for succ in successors {
            self.assign_layers_longest_path(succ, layer + 1);
        }

        // If this is a loop header, process outgoing edges with loop height offset
        {
            let block = self.blocks_by_num.get(&block_num).unwrap();
            if block.attributes.contains(&"loopheader".to_string()) {
                let outgoing_edges = block.outgoing_edges.clone();
                let loop_height = block.loop_height;

                for outgoing_succ in outgoing_edges {
                    self.assign_layers_longest_path(outgoing_succ, layer + loop_height);
                }
            }
        }
    }

    fn assign_layers(&mut self) {
        // TypeScript-style layer assignment with loop hierarchy

        // Find all root blocks (no predecessors) and make them pseudo-loop headers
        let roots: Vec<BlockNumber> = self.blocks.iter()
            .filter(|b| b.predecessors.is_empty())
            .map(|b| b.number)
            .collect();

        // Initialize pseudo-loop headers
        for &root_num in &roots {
            if let Some(root) = self.blocks_by_num.get_mut(&root_num) {
                root.loop_height = 0;
                root.parent_loop = None;
                root.loop_id = root.id; // Root blocks are their own loop
            }
        }

        // Phase 1: Build loop hierarchy
        for &root_num in &roots {
            let root_id = self.blocks_by_num.get(&root_num).unwrap().id;
            self.find_loops(root_num, vec![root_id]);
        }

        // Phase 2: Assign layers using longest-path algorithm
        for &root_num in &roots {
            self.assign_layers_longest_path(root_num, 0);
        }

        // Update the blocks vector with the modified blocks from the maps
        self.blocks = self.blocks_by_num.values().cloned().collect();
        self.blocks.sort_by_key(|b| b.number);

        // Update blocks_by_id
        self.blocks_by_id.clear();
        for block in &self.blocks {
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
        let (nodes_by_layer, _layer_heights, _track_heights) = self.layout();

        // Calculate total SVG dimensions
        let mut max_x: f64 = 0.0;
        let mut max_y: f64 = 0.0;

        for layer in &nodes_by_layer {
            for node in layer {
                max_x = max_x.max(node.pos.x + node.size.x + CONTENT_PADDING);
                max_y = max_y.max(node.pos.y + node.size.y + CONTENT_PADDING);
            }
        }

        let width = (max_x + CONTENT_PADDING) as i32;
        let height = (max_y + CONTENT_PADDING) as i32;

        let mut svg = String::new();
        // Match TypeScript attribute order: xmlns width height viewBox
        svg.push_str(&format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            width, height, width, height));
        svg.push('\n');

        // Add wrapper and styles
        svg.push_str("  <g class=\"ig-graph\">\n");
        svg.push_str("    <rect/>\n");

        // Render blocks
        self.render_blocks_ts_format(&mut svg, &nodes_by_layer);

        // Render arrows
        self.render_arrows_ts_format(&mut svg, &nodes_by_layer);

        svg.push_str("  </g>\n");
        svg.push_str("</svg>");
        svg
    }

    fn render_arrows_ts_format(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        // Build a lookup map for nodes by ID
        let mut node_lookup: HashMap<LayoutNodeID, &LayoutNode> = HashMap::new();
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
                        self.render_single_arrow_ts_format(svg, node, dst_node, port_idx);
                    }
                }
            }
        }
    }

    fn render_single_arrow_ts_format(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode, port_idx: usize) {
        // Calculate arrow start and end points
        let x1 = if src_node.is_dummy() {
            src_node.pos.x
        } else {
            src_node.pos.x + PORT_START + (PORT_SPACING * port_idx as f64).min(src_node.size.x - PORT_START - 10.0)
        };

        let y1 = src_node.pos.y + src_node.size.y;

        let x2 = if dst_node.is_dummy() {
            dst_node.pos.x
        } else {
            dst_node.pos.x + PORT_START
        };

        let y2 = dst_node.pos.y;

        // Get joint offset for this port if available
        let joint_offset = src_node.joint_offsets.get(port_idx).copied().unwrap_or(0.0);

        // Determine if this is a backedge (going backwards in layers)
        let is_backedge = src_node.layer > dst_node.layer;

        // Check if destination is a real block (for arrowhead)
        let do_arrowhead = !dst_node.is_dummy();

        svg.push_str("    <g>\n");

        if !is_backedge {
            // Normal downward arrow
            let ym = y1 + TRACK_PADDING + joint_offset;

            // Check if we need arc-based or bezier-based path
            let horizontal_dist = (x2 - x1).abs();

            if horizontal_dist < 2.0 * ARROW_RADIUS {
                // Narrow arrow - use cubic bezier
                svg.push_str(&format!("      <path d=\"M {} {} C {} {} {} {} {} {}\" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                    x1, y1,
                    x1, y1 + (y2 - y1) / 3.0,
                    x2, y1 + 2.0 * (y2 - y1) / 3.0,
                    x2, y2));
            } else {
                // Wide arrow - use arc+line+arc
                let dir = (x2 - x1).signum();
                let r = ARROW_RADIUS;

                svg.push_str(&format!("      <path d=\"M {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                    x1, y1,
                    x1, ym - r,
                    r, r, if dir > 0.0 { "0" } else { "1" }, x1 + r * dir, ym,
                    x2 - r * dir, ym,
                    r, r, if dir > 0.0 { "0" } else { "1" }, x2, ym + r,
                    x2, y2));
            }

            // Add arrowhead
            if do_arrowhead {
                svg.push_str(&format!("      <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate(180)\"/>\n",
                    x2, y2));
            }
        } else {
            // Backedge arrow - simplified for now
            svg.push_str(&format!("      <path d=\"M {} {} L {} {}\" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x1, y1, x2, y2));

            if do_arrowhead {
                let angle = ((y2 - y1).atan2(x2 - x1) * 180.0 / std::f64::consts::PI) + 180.0;
                svg.push_str(&format!("      <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate({})\"/>\n",
                    x2, y2, angle));
            }
        }

        svg.push_str("    </g>\n");
    }

    fn render_blocks_ts_format(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        for layer in nodes_by_layer {
            for node in layer {
                if node.is_dummy() {
                    continue; // Skip dummy nodes
                }

                if let Some(block) = &node.block {
                    // Start block group with decimal coordinates (no rounding)
                    svg.push_str(&format!("    <g transform=\"translate({}, {})\">\n",
                        node.pos.x, node.pos.y));

                    // Render block rectangles (sizes as integers)
                    svg.push_str(&format!("      <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f9f9f9\" stroke=\"#0c0c0d\" stroke-width=\"1\"/>\n",
                        node.size.x as i32, node.size.y as i32));

                    // Header color
                    let header_color = if block.attributes.contains(&"loopheader".to_string()) {
                        "#1fa411"
                    } else {
                        "#0c0c0d"
                    };

                    svg.push_str(&format!("      <rect x=\"0\" y=\"0\" width=\"{}\" height=\"28\" fill=\"{}\"/>\n",
                        node.size.x as i32, header_color));

                    // Block header text
                    let header_x = node.size.x / 2.0;
                    svg.push_str(&format!("      <text x=\"{}\" y=\"18\" font-family=\"monospace\" font-size=\"12\" fill=\"white\" font-weight=\"bold\" text-anchor=\"middle\">Block {}</text>\n",
                        header_x, block.id.0));

                    // Render instructions with dynamic column positions
                    if let Some(mir_block) = &block.mir_block {
                        // First pass: measure column widths (in pixels, not characters)
                        // IMPORTANT: Measure the RENDERED text (after escape/replace), not the raw JSON text
                        let mut max_num_width: f64 = 0.0;
                        let mut max_opcode_width: f64 = 0.0;
                        let mut max_type_width: f64 = 0.0;

                        for ins in &mir_block.instructions {
                            let num_text = format!("{}", ins.id);
                            max_num_width = max_num_width.max(num_text.chars().count() as f64 * CHAR_WIDTH);

                            // Measure the transformed opcode (after escape and ← replacement)
                            // IMPORTANT: Use .chars().count() not .len() because ← is 3 bytes but 1 character!
                            let opcode_rendered = html_escape(&ins.opcode).replace("&lt;-", "←");
                            max_opcode_width = max_opcode_width.max(opcode_rendered.chars().count() as f64 * CHAR_WIDTH);

                            if ins.instruction_type != "None" {
                                max_type_width = max_type_width.max(ins.instruction_type.chars().count() as f64 * CHAR_WIDTH);
                            }
                        }

                        // Calculate column positions dynamically
                        let num_x = PADDING;
                        let opcode_x = PADDING + max_num_width + 8.0;
                        let type_x = opcode_x + max_opcode_width + 8.0;

                        // Second pass: render instructions
                        let mut y = 40; // Start after header (28px) + padding

                        for ins in &mir_block.instructions {
                            // Determine colors based on attributes
                            let has_movable = ins.attributes.contains(&"Movable".to_string());
                            let has_guard = ins.attributes.contains(&"Guard".to_string());

                            let opcode_color = if has_movable { "#1048af" } else { "black" };
                            let decoration = if has_guard { " text-decoration=\"underline\"" } else { "" };

                            // Escape opcode text and replace <- with ←
                            let opcode = html_escape(&ins.opcode).replace("&lt;-", "←");

                            // Render instruction number at dynamic position
                            svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#777\">{}</text>\n",
                                num_x as i32, y, ins.id));

                            // Render opcode at dynamic position
                            svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"{}\"{}>{}</text>\n",
                                opcode_x as i32, y, opcode_color, decoration, opcode));

                            // Render type if not None at dynamic position
                            if ins.instruction_type != "None" {
                                svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#1048af\">{}</text>\n",
                                    type_x as i32, y, ins.instruction_type));
                            }

                            y += 14; // Move to next line
                        }
                    }

                    // Render branch labels if binary branch
                    if block.successors.len() == 2 {
                        let label_y = node.size.y + 12.0;
                        svg.push_str(&format!("      <text x=\"20\" y=\"{}\" font-family=\"monospace\" font-size=\"9\" fill=\"#777\">1</text>\n", label_y as i32));
                        svg.push_str(&format!("      <text x=\"80\" y=\"{}\" font-family=\"monospace\" font-size=\"9\" fill=\"#777\">0</text>\n", label_y as i32));
                    }

                    svg.push_str("    </g>\n");
                }
            }
        }
    }

    fn render_blocks(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        for layer in nodes_by_layer {
            for node in layer {
                if node.is_dummy() {
                    // Skip rendering dummy nodes - they're just for layout
                    continue;
                } else if let Some(block) = &node.block {
                    // Create a group for this block positioned at its location
                    svg.push_str(&format!(r#"<g transform="translate({}, {})">"#,
                        node.pos.x, node.pos.y));

                    // Determine header color based on attributes
                    let header_color = if block.attributes.contains(&"loopheader".to_string()) {
                        "#1fa411"  // Green for loop headers
                    } else if block.attributes.contains(&"backedge".to_string()) {
                        "#1fa411"  // Green for backedges
                    } else {
                        "#0c0c0d"  // Black for normal blocks
                    };

                    // Render outer rectangle (block border)
                    svg.push_str(&format!(
                        r##"<rect x="0" y="0" width="{}" height="{}" fill="#f9f9f9" stroke="#0c0c0d" stroke-width="1"/>"##,
                        node.size.x, node.size.y
                    ));

                    // Render block header
                    let header_height = 28.0;
                    svg.push_str(&format!(
                        r##"<rect x="0" y="0" width="{}" height="{}" fill="{}"/>"##,
                        node.size.x, header_height, header_color
                    ));

                    // Add block header text
                    let desc = if block.attributes.contains(&"loopheader".to_string()) {
                        " (loop header)"
                    } else if block.attributes.contains(&"backedge".to_string()) {
                        " (backedge)"
                    } else {
                        ""
                    };

                    svg.push_str(&format!(
                        r##"<text x="{}" y="18" font-family="monospace" font-size="12" fill="white" font-weight="bold" text-anchor="middle">Block {}{}</text>"##,
                        node.size.x / 2.0, block.id.0, desc
                    ));

                    // Render MIR instructions if present
                    if let Some(mir_block) = &block.mir_block {
                        let mut y = header_height + 12.0;  // Start below header with padding
                        for ins in &mir_block.instructions {
                            // Render instruction number
                            svg.push_str(&format!(
                                r##"<text x="8" y="{}" font-family="monospace" font-size="11" fill="#777">{}</text>"##,
                                y, ins.id
                            ));

                            // Render opcode
                            let opcode = html_escape(&ins.opcode);
                            svg.push_str(&format!(
                                r##"<text x="23" y="{}" font-family="monospace" font-size="11" fill="black">{}</text>"##,
                                y, opcode
                            ));

                            // Render type (if not None)
                            if ins.instruction_type != "None" {
                                let type_x = node.size.x - 45.0;  // Right-aligned
                                svg.push_str(&format!(
                                    r##"<text x="{}" y="{}" font-family="monospace" font-size="11" fill="#1048af">{}</text>"##,
                                    type_x, y, ins.instruction_type
                                ));
                            }

                            y += 14.0;  // Move down for next instruction
                        }
                    }

                    // Render edge labels for binary branches
                    if block.successors.len() == 2 {
                        let label_y = node.size.y + 12.0;  // Below the block
                        for (i, label) in [1, 0].iter().enumerate() {
                            let label_x = PORT_START + PORT_SPACING * i as f64;
                            svg.push_str(&format!(
                                r##"<text x="{}" y="{}" font-family="monospace" font-size="9" fill="#777">{}</text>"##,
                                label_x, label_y, label
                            ));
                        }
                    }

                    svg.push_str("</g>\n");
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
