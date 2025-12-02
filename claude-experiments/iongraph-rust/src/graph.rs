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
const CHAR_WIDTH: f64 = 7.0; // TypeScript PureSVGTextLayoutProvider uses charWidth = 7
const CHAR_HEIGHT: f64 = 14.0;
const PADDING: f64 = 8.0;
const HEADER_HEIGHT: f64 = 30.0;

// Calculate block size based on instruction content
fn calculate_block_size(block: &Block) -> Vec2 {
    if let Some(mir_block) = &block.mir_block {
        // TypeScript PureSVGTextLayoutProvider formula (lines 263-273):
        // width = padding + maxNumWidth + 8 + maxOpcodeWidth + 8 + maxTypeWidth + padding
        // width = max(headerWidth + padding * 2, calculatedWidth, 150)

        let mut max_opcode_chars = 0;
        let mut max_type_chars = 0;

        for ins in &mir_block.instructions {
            // Count DISPLAY chars (after arrow replacement, before HTML escaping)
            // TypeScript uses textContent.length which counts display chars, not HTML entities
            let opcode_with_arrows = ins.opcode.replace("->", "→").replace("<-", "←");
            max_opcode_chars = max_opcode_chars.max(opcode_with_arrows.chars().count());

            if ins.instruction_type != "None" {
                max_type_chars = max_type_chars.max(ins.instruction_type.chars().count());
            }
        }

        // Calculate column widths
        let max_num_chars = mir_block.instructions.iter()
            .map(|ins| format!("{}", ins.id).len())
            .max()
            .unwrap_or(1);

        let max_num_width = max_num_chars as f64 * CHAR_WIDTH;
        let max_opcode_width = max_opcode_chars as f64 * CHAR_WIDTH;
        let max_type_width = max_type_chars as f64 * CHAR_WIDTH;

        // TypeScript formula: padding + maxNumWidth + 8 + maxOpcodeWidth + 8 + maxTypeWidth + padding
        let calculated_width = PADDING + max_num_width + 8.0 + max_opcode_width + 8.0 + max_type_width + PADDING;

        // Also consider header width (e.g., "Block 12" = 8 chars * 7 = 56)
        let header_text = format!("Block {}", block.number.0);
        let header_width = header_text.len() as f64 * CHAR_WIDTH + PADDING * 2.0;

        // Width is max of header width, calculated width, and absolute minimum 150
        let width = calculated_width.max(header_width).max(150.0);
        let height = HEADER_HEIGHT + (mir_block.instructions.len() as f64 * CHAR_HEIGHT) + PADDING * 2.0;

        Vec2::new(width, height)
    } else if let Some(lir_block) = &block.lir_block {
        // Similar calculation for LIR blocks - use same formula as MIR
        let mut max_opcode_chars = 0;

        for ins in &lir_block.instructions {
            // Count display chars, not HTML entity length
            let opcode_with_arrows = ins.opcode.replace("->", "→").replace("<-", "←");
            max_opcode_chars = max_opcode_chars.max(opcode_with_arrows.chars().count());
        }

        let max_num_chars = lir_block.instructions.iter()
            .map(|ins| format!("{}", ins.id).len())
            .max()
            .unwrap_or(1);

        let max_num_width = max_num_chars as f64 * CHAR_WIDTH;
        let max_opcode_width = max_opcode_chars as f64 * CHAR_WIDTH;

        // LIR blocks don't have types, so: padding + num + 8 + opcode + padding
        let calculated_width = PADDING + max_num_width + 8.0 + max_opcode_width + PADDING;

        let header_text = format!("Block {}", block.number.0);
        let header_width = header_text.len() as f64 * CHAR_WIDTH + PADDING * 2.0;

        let width = calculated_width.max(header_width).max(150.0);
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
}

impl Graph {
    pub fn new(viewport_size: Vec2, pass: Pass) -> Self {
        let mut graph = Self {
            viewport_size,
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
                let block = Block {
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

        // Assign loop_id and loop_num from the stack
        {
            let block = self.blocks_by_num.get_mut(&block_num).unwrap();
            let loop_depth = block.loop_depth as usize;
            if loop_depth < new_loop_ids_by_depth.len() {
                block.loop_id = new_loop_ids_by_depth[loop_depth];
                // Also set loop_num to the block NUMBER of the loop header
                if let Some(loop_header) = self.blocks_by_id.get(&new_loop_ids_by_depth[loop_depth]) {
                    block.loop_num = loop_header.number.0;
                    if block.number.0 == 32 {
                        eprintln!("DEBUG: Block 32 loop_num set to {}, loop_depth={}", block.loop_num, loop_depth);
                    }
                }
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

                    // Get the header's CURRENT layer from blocks_by_num, not the stale blocks_by_id
                    let header_layer = self.blocks_by_num.get(&header_num).unwrap().layer;
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
        let (successors, outgoing, loop_id, _loop_depth) = {
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

        // Track latest backedge dummy for each backedge block (for chaining)
        let mut latest_dummies_for_backedges: HashMap<BlockNumber, LayoutNodeID> = HashMap::new();

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
                            // Update source node's dst_nodes using port index
                            for src_layer_idx in 0..*dummy_layer {
                                if let Some(src_node) = layout_nodes_by_layer[src_layer_idx].iter_mut()
                                    .find(|n| n.id == edge.src_id) {
                                    // Use direct index assignment with port number
                                    if edge.src_port < src_node.dst_nodes.len() {
                                        src_node.dst_nodes[edge.src_port] = Some(dummy_id);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // Create new dummy node
                    let dummy_block = self.blocks_by_num.get(&edge.dst_block_number).unwrap().clone();
                    let mut dummy = LayoutNode::new_dummy_node(node_id, dummy_block.clone(), layer);
                    dummy.src_nodes.push(edge.src_id);

                    dummies_by_dest.insert(edge.dst_block_number, node_id);
                    node_lookup.insert(node_id, layer);

                    // Update source node's dst_nodes using port index
                    for src_layer_idx in 0..layer {
                        if let Some(src_node) = layout_nodes_by_layer[src_layer_idx].iter_mut()
                            .find(|n| n.id == edge.src_id) {
                            // Use direct index assignment with port number
                            if edge.src_port < src_node.dst_nodes.len() {
                                src_node.dst_nodes[edge.src_port] = Some(node_id);
                            }
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
                    if let Some(_dummy) = layout_nodes_by_layer[layer].iter_mut().find(|n| n.id == dummy_id) {
                        // The dummy should eventually connect to the target block
                        // This will be handled in the next layer when the target appears
                    }
                    edge.src_id = dummy_id;
                    edge.src_port = 0;
                }
            }

            // Track which blocks will get backedge dummy nodes
            // (the rightmost block in each active loop at this layer)
            struct PendingLoopDummy {
                loop_id: BlockID,
                block_number: BlockNumber,
            }
            let mut pending_loop_dummies: Vec<PendingLoopDummy> = Vec::new();

            for block in blocks_in_layer {
                // For each block, walk up its loop hierarchy
                if block.loop_id != BlockID(0) {
                    let mut current_loop_id = block.loop_id;
                    loop {
                        // Check if we've seen this loop before
                        if let Some(existing) = pending_loop_dummies.iter_mut().find(|d| d.loop_id == current_loop_id) {
                            // Update to the rightmost block for this loop
                            existing.block_number = block.number;
                        } else {
                            // First block we've seen for this loop
                            pending_loop_dummies.push(PendingLoopDummy {
                                loop_id: current_loop_id,
                                block_number: block.number,
                            });
                        }

                        // Walk to parent loop
                        if let Some(loop_block) = self.blocks_by_id.get(&current_loop_id) {
                            if loop_block.loop_id != BlockID(0) && loop_block.loop_id != current_loop_id {
                                current_loop_id = loop_block.loop_id;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }

            // Track edges to backedge blocks (to be connected after dummies are created)
            let mut backedge_edges: Vec<IncompleteEdge> = Vec::new();

            // Create real nodes for blocks in this layer
            for block in blocks_in_layer {
                let mut layout_node = LayoutNode::new_block_node(node_id, (*block).clone(), layer);

                // Connect terminating edges to this block
                for edge in &terminating_edges {
                    if edge.dst_block_number == block.number {
                        layout_node.src_nodes.push(edge.src_id);
                        // Also update the source node to point to this destination
                        // Find and update the source node's dst_nodes using port index
                        // Include current layer (0..=layer) to handle edges within the same layer
                        for layer_idx in 0..=layer {
                            if let Some(src_node) = layout_nodes_by_layer[layer_idx].iter_mut()
                                .find(|n| n.id == edge.src_id) {
                                // Use direct index assignment with port number
                                if edge.src_port < src_node.dst_nodes.len() {
                                    src_node.dst_nodes[edge.src_port] = Some(node_id);
                                }
                                break;
                            }
                        }
                    }
                }

                // Handle outgoing edges
                let is_backedge_block = block.attributes.contains(&"backedge".to_string());

                if is_backedge_block {
                    // Backedge block: connect directly to loop header (like TypeScript does)
                    for (port, &successor) in block.successors.iter().enumerate() {
                        for target_layer in 0..=layer {
                            if let Some(target_node) = layout_nodes_by_layer[target_layer].iter_mut()
                                .find(|n| n.block.as_ref().map_or(false, |b| b.number == successor)) {
                                if port < layout_node.dst_nodes.len() {
                                    layout_node.dst_nodes[port] = Some(target_node.id);
                                }
                                target_node.src_nodes.push(node_id);
                                break;
                            }
                        }
                    }
                } else {
                    // Regular block: add edges to active edges or defer if targeting backedge
                    for (port, &successor) in block.successors.iter().enumerate() {
                        if let Some(succ_block) = self.blocks_by_num.get(&successor) {
                            if succ_block.attributes.contains(&"backedge".to_string()) {
                                // Defer this edge - will be connected to backedge dummy later
                                backedge_edges.push(IncompleteEdge {
                                    src_id: node_id,
                                    src_port: port,
                                    dst_block_number: successor,
                                });
                            } else if succ_block.layer > layer {
                                // Normal forward edge
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
                let _block_node_id = node_id;
                node_id += 1;

                // Create backedge dummy nodes for this block if it's the rightmost in any loops
                for pending_dummy in &pending_loop_dummies {
                    if pending_dummy.block_number == block.number {
                        // Find the backedge block for this loop
                        if let Some(loop_header) = self.blocks_by_id.get(&pending_dummy.loop_id) {
                            // Find the backedge block that points to this loop header
                            let backedge_block = self.blocks.iter()
                                .find(|b| b.attributes.contains(&"backedge".to_string())
                                    && b.successors.contains(&loop_header.number));

                            if let Some(backedge) = backedge_block {
                                // Create a dummy node for the backedge path
                                let mut backedge_dummy = LayoutNode::new_dummy_node(
                                    node_id,
                                    backedge.clone(),
                                    layer
                                );

                                // Connect this dummy to the previous dummy in the chain (if any)
                                // or directly to the backedge block's layout node
                                if let Some(&prev_dummy_id) = latest_dummies_for_backedges.get(&backedge.number) {
                                    // Chain to previous dummy - new dummy points DOWN to previous dummy
                                    if !backedge_dummy.dst_nodes.is_empty() {
                                        backedge_dummy.dst_nodes[0] = Some(prev_dummy_id);
                                    }
                                    // Update previous dummy to have new dummy as source
                                    for prev_layer in 0..layer {
                                        if let Some(prev_dummy) = layout_nodes_by_layer[prev_layer].iter_mut()
                                            .find(|n| n.id == prev_dummy_id) {
                                            prev_dummy.src_nodes.push(node_id);
                                            break;
                                        }
                                    }
                                } else {
                                    // First dummy - mark for immediate connection and connect to loop header
                                    backedge_dummy.flags |= IMMINENT_BACKEDGE_DUMMY;
                                    // Connect dummy to loop header
                                    for header_layer in 0..=layer {
                                        if let Some(header_node) = layout_nodes_by_layer[header_layer].iter_mut()
                                            .find(|n| n.block.as_ref().map_or(false, |b| b.number == loop_header.number)) {
                                            if !backedge_dummy.dst_nodes.is_empty() {
                                                backedge_dummy.dst_nodes[0] = Some(header_node.id);
                                            }
                                            header_node.src_nodes.push(node_id);
                                            break;
                                        }
                                    }
                                }

                                node_lookup.insert(node_id, layer);
                                layout_nodes_by_layer[layer].push(backedge_dummy);
                                latest_dummies_for_backedges.insert(backedge.number, node_id);
                                node_id += 1;
                            }
                        }
                    }
                }
            }

            // After all block nodes (and their backedge dummies) are created,
            // connect the deferred backedge edges
            for edge in &backedge_edges {
                if let Some(&dummy_id) = latest_dummies_for_backedges.get(&edge.dst_block_number) {
                    // Connect to the latest backedge dummy
                    for layer_idx in 0..=layer {
                        if let Some(src_node) = layout_nodes_by_layer[layer_idx].iter_mut()
                            .find(|n| n.id == edge.src_id) {
                            if edge.src_port < src_node.dst_nodes.len() {
                                src_node.dst_nodes[edge.src_port] = Some(dummy_id);
                            }
                            break;
                        }
                    }

                    // Add src connection to dummy
                    if let Some(dummy) = layout_nodes_by_layer[layer].iter_mut()
                        .find(|n| n.id == dummy_id) {
                        dummy.src_nodes.push(edge.src_id);
                    }
                }
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
        // DEBUG: Log Block 32's initial position and layer
        for (layer_idx, layer) in layout_nodes_by_layer.iter().enumerate() {
            for node in layer.iter() {
                if let Some(block) = &node.block {
                    if block.number.0 == 32 {
                        eprintln!("DEBUG: Block 32 INITIAL position in straighten_edges: layer={}, x={}", layer_idx, node.pos.x);
                    }
                }
            }
            // DEBUG: Show layer 27 node order
            if layer_idx == 27 {
                eprintln!("DEBUG: Layer 27 nodes:");
                for (idx, node) in layer.iter().enumerate() {
                    eprintln!("  [{}] is_dummy={}, block={:?}, x={}", idx, node.is_dummy(),
                        node.block.as_ref().map(|b| b.number), node.pos.x);
                }
            }
        }

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

                if let Some(neighbor_block) = &neighbor.block {
                    if neighbor_block.number.0 == 32 {
                        eprintln!("DEBUG: push_neighbors pushing Block 32: left is_dummy={}, flags={}, at x={}, width={}, node_right_plus_padding={}, Block 32 before={}",
                            node.is_dummy(), node.flags, node.pos.x, node.size.x, node_right_plus_padding, neighbor.pos.x);
                    }
                }

                let old_pos = neighbor.pos.x;
                neighbor.pos.x = neighbor.pos.x.max(node_right_plus_padding);

                if let Some(neighbor_block) = &neighbor.block {
                    if neighbor_block.number.0 == 32 && old_pos != neighbor.pos.x {
                        eprintln!("DEBUG: push_neighbors PUSHED Block 32 from {} to {}", old_pos, neighbor.pos.x);
                    }
                }
            }
        };

        // Push nodes to the right so they fit inside their loop
        // TypeScript iongraph2: pushIntoLoops() - aligns blocks with their SPECIFIC loop header using loopID
        let push_into_loops = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            // First pass: collect loop header positions (block_id -> x position)
            let mut loop_header_positions: HashMap<BlockID, f64> = HashMap::new();
            for layer in nodes_by_layer.iter() {
                for node in layer.iter() {
                    if let Some(block) = &node.block {
                        if block.attributes.contains(&"loopheader".to_string()) {
                            loop_header_positions.insert(block.id, node.pos.x);
                        }
                    }
                }
            }

            // Second pass: apply constraints
            for layer in nodes_by_layer.iter_mut() {
                for node in layer.iter_mut() {
                    if let Some(block) = &node.block {
                        // TypeScript iongraph2: if (node.block.loopID !== null)
                        // In Rust, BlockID(0) is the default/invalid value (like null)
                        if block.loop_id.0 != 0 {
                            // TypeScript iongraph2: const loopHeaderNode = loopHeader.layoutNode;
                            // TypeScript iongraph2: node.pos.x = Math.max(node.pos.x, loopHeaderNode.pos.x);
                            if let Some(&header_x) = loop_header_positions.get(&block.loop_id) {
                                if block.number.0 == 32 {
                                    eprintln!("DEBUG: Block 32 push_into_loops: loop_id={:?}, header_x={}, before pos.x={}", block.loop_id, header_x, node.pos.x);
                                }
                                node.pos.x = node.pos.x.max(header_x);
                                if block.number.0 == 32 {
                                    eprintln!("DEBUG: Block 32 after push_into_loops: pos.x={}", node.pos.x);
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

                // Track the last shifted index to prevent shifting nodes to the left
                let mut last_shifted: i32 = -1;

                // Clone current layer's node data to avoid borrow conflicts
                let current_layer_nodes: Vec<(LayoutNodeID, Vec<Option<LayoutNodeID>>, f64, Option<Block>)> = nodes_by_layer[layer_idx]
                    .iter()
                    .map(|n| (n.id, n.dst_nodes.clone(), n.pos.x, n.block.clone()))
                    .collect();

                for (node_id, dst_nodes, node_x, node_block) in current_layer_nodes {
                    for (src_port, dst_id_opt) in dst_nodes.iter().enumerate() {
                        if let Some(dst_id) = dst_id_opt {
                            // Find destination node in next layer
                            if let Some((dst_idx, dst_node)) = nodes_by_layer[layer_idx + 1]
                                .iter_mut()
                                .enumerate()
                                .find(|(_, n)| n.id == *dst_id) {

                                // Only position if this destination is to the right of the last shifted node
                                if (dst_idx as i32) > last_shifted {
                                    // Check if this is the first parent of the destination
                                    if !dst_node.src_nodes.is_empty() && dst_node.src_nodes[0] == node_id {
                                        let src_port_offset = PORT_START + PORT_SPACING * src_port as f64;
                                        let dst_port_offset = PORT_START;

                                        let x_before = dst_node.pos.x;
                                        let new_x = (node_x + src_port_offset - dst_port_offset).max(dst_node.pos.x);

                                        if let Some(dst_block) = &dst_node.block {
                                            if dst_block.number.0 == 32 {
                                                eprintln!("DEBUG: Block 32 straighten_children: parent Block {:?} at x={}, src_port={}, calculating new_x={} (was {})",
                                                    node_block.as_ref().map(|b| b.number), node_x, src_port, new_x, x_before);
                                            }
                                        }
                                        // DEBUG: Track layer 27 dummy positioning
                                        if dst_node.is_dummy() && layer_idx + 1 == 27 && dst_idx == 0 {
                                            eprintln!("DEBUG: Layer 27 dummy[0] straighten_children: parent Block {:?} at x={}, src_port={}, calculating new_x={} (was {})",
                                                node_block.as_ref().map(|b| b.number), node_x, src_port, new_x, x_before);
                                        }

                                        if new_x != x_before {
                                            dst_node.pos.x = new_x;  // Apply immediately!
                                            last_shifted = dst_idx as i32;
                                            // DEBUG: Track dummy positioning
                                            if dst_node.is_dummy() {
                                                eprintln!("DEBUG: Dummy at idx {} in layer {} positioned from {} to {}", dst_idx, layer_idx + 1, x_before, new_x);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        // Straighten edges that are nearly straight by aligning nodes within threshold
        let straighten_nearly_straight_edges_up = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            for layer_idx in (0..nodes_by_layer.len()).rev() {
                push_neighbors(&mut nodes_by_layer[layer_idx]);

                for node_idx in 0..nodes_by_layer[layer_idx].len() {
                    let src_node_ids: Vec<LayoutNodeID> = nodes_by_layer[layer_idx][node_idx].src_nodes.clone();

                    for &src_id in &src_node_ids {
                        // Find source node (could be in any previous layer)
                        let mut src_is_block = false;
                        let mut src_x = 0.0;

                        for prev_layer in 0..layer_idx {
                            if let Some(src_node) = nodes_by_layer[prev_layer].iter().find(|n| n.id == src_id) {
                                src_is_block = src_node.block.is_some();
                                src_x = src_node.pos.x;
                                break;
                            }
                        }

                        // Only align dummies (skip block-to-block edges)
                        if src_is_block {
                            continue;
                        }

                        let node_x = nodes_by_layer[layer_idx][node_idx].pos.x;
                        let wiggle = (src_x - node_x).abs();

                        if wiggle <= NEARLY_STRAIGHT {
                            // Update both src and dst to align
                            let max_x = src_x.max(node_x);

                            // Update source node
                            for prev_layer in 0..layer_idx {
                                if let Some(src_node) = nodes_by_layer[prev_layer].iter_mut().find(|n| n.id == src_id) {
                                    src_node.pos.x = max_x;
                                    break;
                                }
                            }

                            // Update current node
                            nodes_by_layer[layer_idx][node_idx].pos.x = max_x;
                        }
                    }
                }
            }
        };

        let straighten_nearly_straight_edges_down = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            for layer_idx in 0..nodes_by_layer.len() {
                push_neighbors(&mut nodes_by_layer[layer_idx]);

                for node_idx in 0..nodes_by_layer[layer_idx].len() {
                    let dst_nodes = nodes_by_layer[layer_idx][node_idx].dst_nodes.clone();

                    if dst_nodes.is_empty() {
                        continue;
                    }

                    if let Some(Some(dst_id)) = dst_nodes.get(0) {
                        // Find destination node in next layer
                        if layer_idx + 1 < nodes_by_layer.len() {
                            if let Some(dst_node) = nodes_by_layer[layer_idx + 1].iter().find(|n| n.id == *dst_id) {
                                // Only align dummies
                                if dst_node.block.is_some() {
                                    continue;
                                }

                                let node_x = nodes_by_layer[layer_idx][node_idx].pos.x;
                                let dst_x = dst_node.pos.x;
                                let wiggle = (dst_x - node_x).abs();

                                if wiggle <= NEARLY_STRAIGHT {
                                    let max_x = dst_x.max(node_x);

                                    // Update both nodes
                                    nodes_by_layer[layer_idx][node_idx].pos.x = max_x;

                                    if let Some(dst_node_mut) = nodes_by_layer[layer_idx + 1].iter_mut().find(|n| n.id == *dst_id) {
                                        dst_node_mut.pos.x = max_x;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        // The main pass sequence (matching TypeScript exactly)
        // First: layoutIterations x (straightenChildren, pushIntoLoops, straightenDummyRuns)
        for _ in 0..LAYOUT_ITERATIONS {
            straighten_children(layout_nodes_by_layer);
            push_into_loops(layout_nodes_by_layer);
            straighten_dummy_runs(layout_nodes_by_layer);
        }

        straighten_dummy_runs(layout_nodes_by_layer);

        // Conservative straightening: align nodes with parents/children without causing overlap
        let straighten_conservative = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            // Build a position lookup first to avoid borrow issues
            let mut node_positions: HashMap<LayoutNodeID, f64> = HashMap::new();
            for layer in nodes_by_layer.iter() {
                for node in layer {
                    node_positions.insert(node.id, node.pos.x);
                }
            }

            for layer_idx in 0..nodes_by_layer.len() {
                let layer_len = nodes_by_layer[layer_idx].len();

                // Walk right to left
                for i in (0..layer_len).rev() {
                    // Only process block nodes, not backedges
                    let is_block = nodes_by_layer[layer_idx][i].block.is_some();
                    let is_backedge = nodes_by_layer[layer_idx][i].block.as_ref()
                        .map_or(false, |b| b.attributes.contains(&"backedge".to_string()));

                    if !is_block || is_backedge {
                        continue;
                    }

                    let current_node_id = nodes_by_layer[layer_idx][i].id;
                    let current_node_x = nodes_by_layer[layer_idx][i].pos.x;
                    let mut deltas_to_try: Vec<f64> = Vec::new();

                    // Calculate deltas to align with parents
                    let src_nodes = nodes_by_layer[layer_idx][i].src_nodes.clone();
                    for &parent_id in &src_nodes {
                        let parent_x = node_positions.get(&parent_id).copied().unwrap_or(0.0);

                        // Find which port connects to current node
                        let mut port_in_parent = 0;
                        'find_port: for layer in nodes_by_layer.iter() {
                            for node in layer {
                                if node.id == parent_id {
                                    for (port, dst_opt) in node.dst_nodes.iter().enumerate() {
                                        if let Some(dst_id) = dst_opt {
                                            if *dst_id == current_node_id {
                                                port_in_parent = port;
                                                break 'find_port;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let src_port_offset = PORT_START + port_in_parent as f64 * PORT_SPACING;
                        let dst_port_offset = PORT_START;
                        let delta = (parent_x + src_port_offset) - (current_node_x + dst_port_offset);
                        deltas_to_try.push(delta);
                    }

                    // Calculate deltas to align with children
                    let dst_nodes = nodes_by_layer[layer_idx][i].dst_nodes.clone();
                    for (src_port, dst_id_opt) in dst_nodes.iter().enumerate() {
                        if let Some(dst_id) = dst_id_opt {
                            let dst_x = node_positions.get(dst_id).copied().unwrap_or(0.0);

                            // Check if it's a backedge dummy
                            let mut is_backedge_dummy = false;
                            for layer in nodes_by_layer.iter() {
                                for node in layer {
                                    if node.id == *dst_id {
                                        is_backedge_dummy = node.is_dummy() &&
                                            node.dst_block.as_ref().map_or(false, |b| b.attributes.contains(&"backedge".to_string()));
                                        break;
                                    }
                                }
                            }

                            if is_backedge_dummy {
                                continue;
                            }

                            let src_port_offset = PORT_START + src_port as f64 * PORT_SPACING;
                            let dst_port_offset = PORT_START;
                            let delta = (dst_x + dst_port_offset) - (current_node_x + src_port_offset);
                            deltas_to_try.push(delta);
                        }
                    }

                    // If already aligned (delta == 0), skip
                    if deltas_to_try.iter().any(|&d| d.abs() < 0.1) {
                        continue;
                    }

                    // Filter to positive deltas and sort
                    deltas_to_try.retain(|&d| d > 0.0);
                    deltas_to_try.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Try each delta, taking the first that doesn't cause overlap
                    for delta in deltas_to_try {
                        let mut overlaps_any = false;

                        // Check for overlap with nodes to the right
                        for j in (i + 1)..layer_len {
                            // Ignore rightmost dummies
                            if (nodes_by_layer[layer_idx][j].flags & RIGHTMOST_DUMMY) != 0 {
                                continue;
                            }

                            let a1 = current_node_x + delta;
                            let a2 = a1 + nodes_by_layer[layer_idx][i].size.x;
                            let b1 = nodes_by_layer[layer_idx][j].pos.x - BLOCK_GAP;
                            let b2 = nodes_by_layer[layer_idx][j].pos.x + nodes_by_layer[layer_idx][j].size.x + BLOCK_GAP;

                            if a2 >= b1 && a1 <= b2 {
                                overlaps_any = true;
                                break;
                            }
                        }

                        if !overlaps_any {
                            nodes_by_layer[layer_idx][i].pos.x += delta;
                            node_positions.insert(current_node_id, nodes_by_layer[layer_idx][i].pos.x);
                            break;
                        }
                    }
                }

                push_neighbors(&mut nodes_by_layer[layer_idx]);
            }
        };

        // Second: nearlyStraightIterations x (straightenNearlyStraightEdgesUp, Down)
        for _ in 0..NEARLY_STRAIGHT_ITERATIONS {
            straighten_nearly_straight_edges_up(layout_nodes_by_layer);
            straighten_nearly_straight_edges_down(layout_nodes_by_layer);
        }

        // Suck in leftmost dummies: pull them right without exceeding parent/destination positions
        let suck_in_leftmost_dummies = |nodes_by_layer: &mut Vec<Vec<LayoutNode>>| {
            // Pre-build lookup maps to avoid borrow issues
            let mut node_positions: HashMap<LayoutNodeID, f64> = HashMap::new();
            let mut node_to_port_map: HashMap<(LayoutNodeID, LayoutNodeID), usize> = HashMap::new(); // (src_id, dst_id) -> port
            let mut block_positions: HashMap<BlockNumber, f64> = HashMap::new();

            for layer in nodes_by_layer.iter() {
                for node in layer {
                    node_positions.insert(node.id, node.pos.x);

                    if let Some(block) = &node.block {
                        block_positions.insert(block.number, node.pos.x);
                    }

                    // Map dst nodes to ports
                    for (port, dst_opt) in node.dst_nodes.iter().enumerate() {
                        if let Some(dst_id) = dst_opt {
                            node_to_port_map.insert((node.id, *dst_id), port);
                        }
                    }
                }
            }

            let mut dummy_run_positions: HashMap<BlockNumber, f64> = HashMap::new();

            for layer_idx in 0..nodes_by_layer.len() {
                // Find leftmost non-dummy node
                let mut first_non_dummy_idx = nodes_by_layer[layer_idx].len();
                let mut next_x = 0.0;

                for (i, node) in nodes_by_layer[layer_idx].iter().enumerate() {
                    if (node.flags & LEFTMOST_DUMMY) == 0 {
                        first_non_dummy_idx = i;
                        next_x = node.pos.x;
                        break;
                    }
                }

                // Walk backward through leftmost dummies
                if first_non_dummy_idx > 0 {
                    next_x -= BLOCK_GAP + PORT_START;

                    for i in (0..first_non_dummy_idx).rev() {
                        let mut max_safe_x = next_x;

                        // Check parent constraints
                        let src_node_ids = nodes_by_layer[layer_idx][i].src_nodes.clone();
                        let dummy_id = nodes_by_layer[layer_idx][i].id;

                        for &src_id in &src_node_ids {
                            let src_x = node_positions.get(&src_id).copied().unwrap_or(0.0);
                            let port_idx = node_to_port_map.get(&(src_id, dummy_id)).copied().unwrap_or(0);

                            let src_x_with_port = src_x + (port_idx as f64 * PORT_SPACING);
                            if src_x_with_port < max_safe_x {
                                max_safe_x = src_x_with_port;
                            }
                        }

                        // Check destination block constraint
                        if let Some(dst_block) = &nodes_by_layer[layer_idx][i].dst_block {
                            if let Some(&block_x) = block_positions.get(&dst_block.number) {
                                if block_x < max_safe_x {
                                    max_safe_x = block_x;
                                }
                            }

                            // Track minimum position for this backedge run
                            let current_min = dummy_run_positions.get(&dst_block.number).copied().unwrap_or(f64::INFINITY);
                            dummy_run_positions.insert(dst_block.number, current_min.min(max_safe_x));
                        }

                        nodes_by_layer[layer_idx][i].pos.x = max_safe_x;
                        node_positions.insert(dummy_id, max_safe_x);
                        next_x = max_safe_x - BLOCK_GAP;
                    }
                }
            }

            // Apply minimum positions to all leftmost dummies in each run
            for layer in nodes_by_layer.iter_mut() {
                for node in layer.iter_mut() {
                    if (node.flags & LEFTMOST_DUMMY) != 0 {
                        if let Some(dst_block) = &node.dst_block {
                            if let Some(&min_x) = dummy_run_positions.get(&dst_block.number) {
                                node.pos.x = min_x;
                            }
                        }
                    }
                }
            }
        };

        // Third: straightenConservative
        straighten_conservative(layout_nodes_by_layer);
        straighten_dummy_runs(layout_nodes_by_layer);

        // Fourth: suckInLeftmostDummies
        suck_in_leftmost_dummies(layout_nodes_by_layer);
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

            // Collect next layer node IDs for filtering forward edges
            let next_layer_ids: std::collections::HashSet<LayoutNodeID> = if layer_idx + 1 < layout_nodes_by_layer.len() {
                layout_nodes_by_layer[layer_idx + 1].iter().map(|n| n.id).collect()
            } else {
                std::collections::HashSet::new()
            };

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

                for (src_port, dst_id_opt) in node.dst_nodes.iter().enumerate() {
                    if let Some(dst_id) = dst_id_opt {
                        // Only create joints for forward edges to the next layer
                        // Skip backedges or edges to other layers
                        if !next_layer_ids.contains(dst_id) {
                            continue;
                        }

                        let dst_x = node_positions.get(dst_id).copied()
                            .expect(&format!("Failed to find destination node {} in positions map", dst_id));

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
            }

            joints.sort_by(|a, b| a.x1.partial_cmp(&b.x1).unwrap());

            if layer_idx == 26 {
                eprintln!("DEBUG layer 26 nodes:");
                for (i, node) in layout_nodes_by_layer[26].iter().enumerate() {
                    eprintln!("  [{}] id={}, x={}, is_dummy={}, block={:?}",
                        i, node.id, node.pos.x, node.is_dummy(), node.block.as_ref().map(|b| b.number));
                }
                eprintln!("DEBUG layer 27 nodes:");
                for (i, node) in layout_nodes_by_layer[27].iter().enumerate() {
                    eprintln!("  [{}] id={}, x={}, is_dummy={}, block={:?}",
                        i, node.id, node.pos.x, node.is_dummy(), node.block.as_ref().map(|b| b.number));
                }
                eprintln!("DEBUG finangle_joints: layer 26 has {} joints", joints.len());
                for (i, joint) in joints.iter().enumerate() {
                    eprintln!("  joint {}: x1={}, x2={}, src_id={}, dst_id={}, direction={}",
                        i, joint.x1, joint.x2, joint.src_id, joint.dst_id,
                        if joint.x2 - joint.x1 >= 0.0 { "rightward" } else { "leftward" });
                }
            }

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

            if layer_idx == 26 {
                eprintln!("DEBUG finangle_joints: layer 26 - rightward_tracks.len()={}, leftward_tracks.len()={}, tracks_height={}",
                    rightward_tracks.len(), leftward_tracks.len(), tracks_height);
                eprintln!("  rightward_tracks: {:?}", rightward_tracks.iter().map(|t| t.len()).collect::<Vec<_>>());
                eprintln!("  leftward_tracks: {:?}", leftward_tracks.iter().map(|t| t.len()).collect::<Vec<_>>());
            }

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
                if let Some(block) = &node.block {
                    if block.number.0 == 32 {
                        eprintln!("DEBUG verticalize: Block 32 at layer {} set to y={}", i, next_layer_y);
                    }
                }
            }

            // Layer height is just the maximum node height (track height is added separately)
            let track_height = track_heights.get(i).copied().unwrap_or(0.0);
            let layer_height = max_node_height;
            layer_heights[i] = layer_height;

            if i >= 25 && i <= 28 {
                eprintln!("DEBUG verticalize: layer {} - max_node_height={}, track_height={}, layer_height={}, next_layer_y before={}, after={}",
                    i, max_node_height, track_height, layer_height, next_layer_y, next_layer_y + layer_height + TRACK_PADDING + track_height + TRACK_PADDING);
            }

            next_layer_y += layer_height + TRACK_PADDING + track_height + TRACK_PADDING;
        }

        layer_heights
    }

    // SVG rendering methods
    pub fn render_svg(&mut self) -> String {
        let (nodes_by_layer, layer_heights, track_heights) = self.layout();

        // Calculate total SVG dimensions
        let mut max_x: f64 = 0.0;
        let mut max_y: f64 = 0.0;

        for layer in &nodes_by_layer {
            for node in layer {
                max_x = max_x.max(node.pos.x + node.size.x + CONTENT_PADDING);
                max_y = max_y.max(node.pos.y + node.size.y + CONTENT_PADDING);
            }
        }

        // TypeScript: createRenderingSurface gets maxX/maxY, then adds 40 in generate script
        // See generate-svg-function.mjs lines 32-33: width = Math.ceil(graphSize.x + 40)
        let width = (max_x + 40.0).ceil() as i32;
        let height = (max_y + 40.0).ceil() as i32;

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
        self.render_arrows_ts_format(&mut svg, &nodes_by_layer, &layer_heights, &track_heights);

        svg.push_str("  </g>\n");
        svg.push_str("</svg>");
        svg
    }

    fn render_arrows_ts_format(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>], layer_heights: &[f64], track_heights: &[f64]) {
        // Build a lookup map for nodes by ID
        let mut node_lookup: HashMap<LayoutNodeID, &LayoutNode> = HashMap::new();
        for layer in nodes_by_layer {
            for node in layer {
                node_lookup.insert(node.id, node);
            }
        }

        // Track if we render any arrows (TypeScript uses <g/> if empty)
        let mut has_arrows = false;
        let mut arrows_content = String::new();

        // Render arrows for all nodes
        for (layer_idx, layer) in nodes_by_layer.iter().enumerate() {
            for node in layer {
                // Check for special arrow types first (like TypeScript does)

                // 1. Backedge block - renders horizontal loop header arrow
                if let Some(block) = &node.block {
                    if block.attributes.contains(&"backedge".to_string()) {
                        // Render loop header arrow (handled specially)
                        if let Some(&dst_id) = node.dst_nodes.get(0).and_then(|id| id.as_ref()) {
                            if let Some(dst_node) = node_lookup.get(&dst_id) {
                                self.render_loop_header_arrow(&mut arrows_content, node, dst_node);
                                has_arrows = true;
                            }
                        }
                        continue; // Skip normal arrow rendering for backedge blocks
                    }
                }

                // 2. IMMINENT_BACKEDGE_DUMMY - renders arrow to backedge block
                if (node.flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
                    // Find the backedge block's layout node
                    if let Some(backedge_block) = &node.dst_block {
                        // Find the layout node for the backedge block
                        if let Some(backedge_layout_node) = nodes_by_layer.iter()
                            .flat_map(|layer| layer.iter())
                            .find(|n| n.block.as_ref().map_or(false, |b| b.number == backedge_block.number)) {
                            self.render_arrow_to_backedge(&mut arrows_content, node, backedge_layout_node);
                            has_arrows = true;
                        }
                    }
                    continue; // Skip normal arrow rendering
                }

                // 3. Check for arrows to backedge dummies (TypeScript line 1150)
                for (port_idx, dst_id_opt) in node.dst_nodes.iter().enumerate() {
                    if let Some(dst_id) = dst_id_opt {
                        if let Some(dst_node) = node_lookup.get(dst_id) {
                            // Check if destination is a backedge dummy
                            let is_backedge_dummy = dst_node.is_dummy() &&
                                dst_node.dst_block.as_ref()
                                    .map_or(false, |b| b.attributes.contains(&"backedge".to_string()));

                            if is_backedge_dummy {
                                // Special handling for arrows to backedge dummies
                                if node.is_dummy() {
                                    // Dummy-to-dummy: draw upward arrow
                                    self.render_upward_arrow_between_dummies(&mut arrows_content, node, dst_node, port_idx);
                                    has_arrows = true;
                                } else {
                                    // Block-to-dummy: draw arrow from block to backedge dummy
                                    self.render_arrow_to_backedge_dummy(&mut arrows_content, node, dst_node, port_idx, layer_idx, layer_heights, track_heights);
                                    has_arrows = true;
                                }
                            } else {
                                // Normal arrow
                                self.render_single_arrow_ts_format(&mut arrows_content, node, dst_node, port_idx, layer_idx, layer_heights, track_heights);
                                has_arrows = true;
                            }
                        }
                    }
                }
            }
        }

        // Emit <g/> if no arrows, otherwise <g>content</g> (TypeScript behavior)
        if has_arrows {
            svg.push_str("    <g>\n");
            svg.push_str(&arrows_content);
            svg.push_str("    </g>\n");
        } else {
            svg.push_str("    <g/>\n");
        }
    }

    fn render_single_arrow_ts_format(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode, port_idx: usize, layer_idx: usize, layer_heights: &[f64], track_heights: &[f64]) {
        // Calculate arrow start and end points
        let mut x1 = if src_node.is_dummy() {
            src_node.pos.x
        } else {
            src_node.pos.x + PORT_START + (PORT_SPACING * port_idx as f64).min(src_node.size.x - PORT_START - 10.0)
        };

        let y1 = src_node.pos.y + src_node.size.y;

        let mut x2 = if dst_node.is_dummy() {
            dst_node.pos.x + PORT_START  // Dummies also use PORT_START for arrow rendering
        } else {
            dst_node.pos.x + PORT_START
        };

        // Check if destination is a backedge dummy
        let is_backedge_dummy = dst_node.is_dummy() &&
            dst_node.dst_block.as_ref()
                .map_or(false, |b| b.attributes.contains(&"backedge".to_string()));

        // Y2 coordinate depends on whether it's an IMMINENT_BACKEDGE_DUMMY
        let y2 = if is_backedge_dummy && (dst_node.flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
            dst_node.pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS
        } else {
            dst_node.pos.y
        };

        // Get joint offset for this port if available
        let joint_offset = src_node.joint_offsets.get(port_idx).copied().unwrap_or(0.0);

        // Determine if this is a backedge (going backwards in layers)
        let is_backedge = src_node.layer > dst_node.layer;

        // Check if destination is a real block (for arrowhead)
        let do_arrowhead = !dst_node.is_dummy();

        // Align stroke to pixels (for crisp 1px lines)
        let stroke = 1;
        if stroke % 2 == 1 {
            x1 += 0.5;
            x2 += 0.5;
        }

        svg.push_str("      <g>\n");

        if !is_backedge {
            // Normal downward arrow
            // TypeScript: const ym = (y1 - node.size.y) + layerHeights[layer] + TRACK_PADDING + trackHeights[layer] / 2 + node.jointOffsets[i];
            let layer_height = layer_heights.get(layer_idx).copied().unwrap_or(0.0);
            let track_height = track_heights.get(layer_idx).copied().unwrap_or(0.0);
            let mut ym = (y1 - src_node.size.y) + layer_height + TRACK_PADDING + track_height / 2.0 + joint_offset;

            // Align ym coordinate for crisp rendering
            if stroke % 2 == 1 {
                ym += 0.5;
            }

            // Check if we need arc-based or bezier-based path
            let horizontal_dist = (x2 - x1).abs();

            if horizontal_dist < 2.0 * ARROW_RADIUS {
                // Narrow arrow - use cubic bezier with destination x-coordinate throughout
                // This makes the arrow perfectly vertical aligned with the destination
                svg.push_str(&format!("        <path d=\"M {} {} C {} {} {} {} {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                    x2, y1,
                    x2, y1 + (y2 - y1) / 3.0,
                    x2, y1 + 2.0 * (y2 - y1) / 3.0,
                    x2, y2));
            } else {
                // Wide arrow - use arc+line+arc
                let dir = (x2 - x1).signum();
                let r = ARROW_RADIUS;

                svg.push_str(&format!("        <path d=\"M {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                    x1, y1,
                    x1, ym - r,
                    r, r, if dir > 0.0 { "0" } else { "1" }, x1 + r * dir, ym,
                    x2 - r * dir, ym,
                    r, r, if dir > 0.0 { "1" } else { "0" }, x2, ym + r,
                    x2, y2));
            }

            // Add arrowhead
            if do_arrowhead {
                svg.push_str(&format!("        <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate(180)\"/>\n",
                    x2, y2));
            }
        } else {
            // Backedge arrow - use cubic bezier for vertical backedge returns
            // TypeScript: path += `C ${x1} ${y1 + (y2 - y1) / 3} ${x2} ${y1 + 2 * (y2 - y1) / 3} ${x2} ${y2} `;
            svg.push_str(&format!("        <path d=\"M {} {} C {} {} {} {} {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x1, y1,
                x1, y1 + (y2 - y1) / 3.0,
                x2, y1 + 2.0 * (y2 - y1) / 3.0,
                x2, y2));

            if do_arrowhead {
                let angle = ((y2 - y1).atan2(x2 - x1) * 180.0 / std::f64::consts::PI) + 180.0;
                svg.push_str(&format!("        <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate({})\"/>\n",
                    x2, y2, angle));
            }
        }

        svg.push_str("      </g>\n");
    }

    fn render_loop_header_arrow(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode) {
        // Horizontal arrow from backedge block (src_node) to loop header (dst_node)
        // Arrow goes LEFT from backedge to loop header
        // TypeScript: const x1 = node.pos.x;  (backedge block left edge)
        // TypeScript: const y1 = node.pos.y + HEADER_ARROW_PUSHDOWN;
        // TypeScript: const x2 = header.layoutNode.pos.x + header.size.x;  (loop header right edge)
        // TypeScript: const y2 = header.layoutNode.pos.y + HEADER_ARROW_PUSHDOWN;

        let x1 = src_node.pos.x;  // Backedge block left edge
        let y1 = src_node.pos.y + HEADER_ARROW_PUSHDOWN;
        let x2 = dst_node.pos.x + dst_node.size.x;  // Loop header right edge
        let _y2 = dst_node.pos.y + HEADER_ARROW_PUSHDOWN;

        // Pixel alignment for crisp 1px strokes
        let y_aligned = y1 + 0.5;

        svg.push_str("      <g>\n");
        svg.push_str(&format!("        <path d=\"M {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
            x1, y_aligned, x2, y_aligned));
        // Arrowhead pointing left
        svg.push_str(&format!("        <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate(270)\"/>\n",
            x2, y_aligned));
        svg.push_str("      </g>\n");
    }

    fn render_arrow_to_backedge(&self, svg: &mut String, src_node: &LayoutNode, backedge_node: &LayoutNode) {
        // Arrow from IMMINENT_BACKEDGE_DUMMY to backedge block
        // TypeScript: const x1 = node.pos.x + PORT_START;
        // TypeScript: const y1 = node.pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS;
        // TypeScript: const x2 = backedge.layoutNode.pos.x + backedge.size.x;
        // TypeScript: const y2 = backedge.layoutNode.pos.y + HEADER_ARROW_PUSHDOWN;

        let mut x1 = src_node.pos.x + PORT_START;
        let y1 = src_node.pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS;
        let x2 = backedge_node.pos.x + backedge_node.size.x;  // No pixel alignment for x2
        let y2 = backedge_node.pos.y + HEADER_ARROW_PUSHDOWN;

        // Pixel alignment for x1 only
        x1 += 0.5;

        let r = ARROW_RADIUS;
        let mut ym = (y1 - r).max(y2);

        // Pixel alignment for ym
        ym += 0.5;

        svg.push_str("      <g>\n");

        // Path with arc routing - goes directly from start to arc, no initial line segment
        svg.push_str(&format!("        <path d=\"M {} {} A {} {} 0 0 0 {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
            x1, y1,
            r, r,
            x1 - r, ym,
            x2, ym));

        // Arrowhead pointing left
        svg.push_str(&format!("        <path d=\"M 0 0 L -5 7.5 L 5 7.5 Z\" transform=\"translate({}, {}) rotate(270)\"/>\n",
            x2, ym));

        svg.push_str("      </g>\n");
    }

    fn render_upward_arrow_between_dummies(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode, port_idx: usize) {
        // TypeScript line 1154-1156: Draw upward arrow between dummies
        // const ym = y1 - TRACK_PADDING; // this really shouldn't matter because we should straighten all these out
        // this.layoutProvider.drawUpwardArrow(x1, y1, x2, y2, ym, false);

        let mut x1 = src_node.pos.x + PORT_START + (PORT_SPACING * port_idx as f64);
        let y1 = src_node.pos.y + src_node.size.y;
        let mut x2 = dst_node.pos.x + PORT_START;
        let y2 = if (dst_node.flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
            dst_node.pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS
        } else {
            dst_node.pos.y
        };

        let mut ym = y1 - TRACK_PADDING;

        // Pixel alignment for crisp 1px strokes
        x1 += 0.5;
        x2 += 0.5;
        ym += 0.5;

        svg.push_str("      <g>\n");

        // Match TypeScript createUpwardArrowSVG (line 415-433)
        let r = ARROW_RADIUS;
        let horizontal_dist = (x2 - x1).abs();

        if horizontal_dist < 2.0 * r {
            // Narrow arrow - use cubic bezier with destination x-coordinate (TypeScript line 422)
            svg.push_str(&format!("        <path d=\"M {} {} C {} {} {} {} {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x2, y1,
                x2, y1 + (y2 - y1) / 3.0,
                x2, y1 + 2.0 * (y2 - y1) / 3.0,
                x2, y2));
        } else {
            // Wide arrow - use arc routing (TypeScript line 425)
            let dir = (x2 - x1).signum();
            svg.push_str(&format!("        <path d=\"M {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x1, y1,
                x1, ym + r,
                r, r, if dir > 0.0 { "1" } else { "0" }, x1 + r * dir, ym,
                x2 - r * dir, ym,
                r, r, if dir > 0.0 { "0" } else { "1" }, x2, ym - r,
                x2, y2));
        }

        // No arrowhead for dummy-to-dummy (hasArrowhead = false in TypeScript)
        svg.push_str("      </g>\n");
    }

    fn render_arrow_to_backedge_dummy(&self, svg: &mut String, src_node: &LayoutNode, dst_node: &LayoutNode, port_idx: usize, layer_idx: usize, layer_heights: &[f64], track_heights: &[f64]) {
        // TypeScript line 1158-1160: Draw arrow from block to backedge dummy
        // const ym = (y1 - node.size.y) + layerHeights[layer] + TRACK_PADDING + trackHeights[layer] / 2 + node.jointOffsets[i];
        // this.layoutProvider.drawArrowFromBlockToBackedgeDummy(x1, y1, x2, y2, ym);

        let mut x1 = src_node.pos.x + PORT_START + (PORT_SPACING * port_idx as f64).min(src_node.size.x - PORT_START - 10.0);
        let y1 = src_node.pos.y + src_node.size.y;
        let mut x2 = dst_node.pos.x + PORT_START;
        let y2 = if (dst_node.flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
            dst_node.pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS
        } else {
            dst_node.pos.y
        };

        let joint_offset = src_node.joint_offsets.get(port_idx).copied().unwrap_or(0.0);
        let layer_height = layer_heights.get(layer_idx).copied().unwrap_or(0.0);
        let track_height = track_heights.get(layer_idx).copied().unwrap_or(0.0);
        let mut ym = (y1 - src_node.size.y) + layer_height + TRACK_PADDING + track_height / 2.0 + joint_offset;

        // Pixel alignment for crisp 1px strokes
        x1 += 0.5;
        x2 += 0.5;
        ym += 0.5;

        svg.push_str("      <g>\n");

        // Draw path with arc routing
        // Note: When ym > y1, the arrow makes a downward detour before going up
        // TypeScript: L x1 ym-r, arc to ym, horizontal, arc to ym-r, L x2 y2
        let r = ARROW_RADIUS;
        let horizontal_dist = (x2 - x1).abs();

        if horizontal_dist < 2.0 * r {
            // Narrow arrow - use cubic bezier with destination x-coordinate
            svg.push_str(&format!("        <path d=\"M {} {} C {} {} {} {} {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x2, y1, x2, ym, x2, ym, x2, y2));
        } else {
            // Wide arrow - use arc routing with ym-r for both vertical segments
            // This creates a downward detour when ym > y1
            let dir = (x2 - x1).signum();
            svg.push_str(&format!("        <path d=\"M {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} A {} {} 0 0 {} {} {} L {} {} \" fill=\"none\" stroke=\"black\" stroke-width=\"1 \"/>\n",
                x1, y1,
                x1, ym - r,
                r, r, if dir > 0.0 { "0" } else { "1" }, x1 + r * dir, ym,
                x2 - r * dir, ym,
                r, r, if dir > 0.0 { "0" } else { "1" }, x2, ym - r,
                x2, y2));
        }

        // No arrowhead for arrows to backedge dummies
        svg.push_str("      </g>\n");
    }

    fn render_blocks_ts_format(&self, svg: &mut String, nodes_by_layer: &[Vec<LayoutNode>]) {
        // Collect all non-dummy nodes from all layers
        let mut all_nodes: Vec<&LayoutNode> = nodes_by_layer
            .iter()
            .flat_map(|layer| layer.iter())
            .filter(|node| !node.is_dummy())
            .collect();

        // Sort by block ID globally (matching TypeScript behavior)
        all_nodes.sort_by_key(|node| {
            node.block.as_ref().map(|b| b.id.0).unwrap_or(u32::MAX)
        });

        // Render blocks in sorted order
        for node in all_nodes {

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
                    let header_label = if block.attributes.contains(&"loopheader".to_string()) {
                        format!("Block {} (loop header)", block.id.0)
                    } else if block.attributes.contains(&"backedge".to_string()) {
                        format!("Block {} (backedge)", block.id.0)
                    } else {
                        format!("Block {}", block.id.0)
                    };
                    svg.push_str(&format!("      <text x=\"{}\" y=\"18\" font-family=\"monospace\" font-size=\"12\" fill=\"white\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>\n",
                        header_x, header_label));

                    // Render instructions with dynamic column positions (matching TypeScript)
                    if let Some(mir_block) = &block.mir_block {
                        // Calculate column positions using same formula as calculate_block_size
                        let mut max_opcode_chars = 0;
                        let max_num_chars = mir_block.instructions.iter()
                            .map(|ins| format!("{}", ins.id).len())
                            .max()
                            .unwrap_or(1);

                        for ins in &mir_block.instructions {
                            // Must match calculate_block_size logic - count display chars, not HTML entities!
                            let opcode_with_arrows = ins.opcode.replace("->", "→").replace("<-", "←");
                            max_opcode_chars = max_opcode_chars.max(opcode_with_arrows.chars().count());
                        }

                        // Column positions matching TypeScript's formula
                        let num_x = 8.0;
                        let opcode_x = 8.0 + (max_num_chars as f64 * CHAR_WIDTH) + 8.0;
                        let type_x = opcode_x + (max_opcode_chars as f64 * CHAR_WIDTH) + 8.0;

                        // Second pass: render instructions
                        let mut y = 40; // Start after header (28px) + padding

                        for ins in &mir_block.instructions {
                            // Determine colors based on attributes
                            let has_movable = ins.attributes.contains(&"Movable".to_string());
                            let has_guard = ins.attributes.contains(&"Guard".to_string());

                            let opcode_color = if has_movable { "#1048af" } else { "black" };
                            let decoration = if has_guard { " text-decoration=\"underline\"" } else { "" };

                            // Replace ASCII arrows with Unicode before HTML-escaping
                            let opcode_with_arrows = ins.opcode.replace("->", "→").replace("<-", "←");
                            let opcode = html_escape(&opcode_with_arrows);

                            // Render instruction number at dynamic position
                            svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#777\">{}</text>\n",
                                num_x, y, ins.id));

                            // Render opcode at dynamic position
                            svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"{}\"{}>{}</text>\n",
                                opcode_x, y, opcode_color, decoration, opcode));

                            // Render type if not None at dynamic position
                            if ins.instruction_type != "None" {
                                svg.push_str(&format!("      <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#1048af\">{}</text>\n",
                                    type_x, y, ins.instruction_type));
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
        assert!(svg.contains("Block 0"));
        assert!(svg.contains("Block 1"));
    }
    
    #[test]
    fn test_svg_rendering_complex() {
        let pass = create_complex_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        let svg = graph.render_svg();

        // Should contain all 5 blocks
        assert!(svg.contains("Block 0"));
        assert!(svg.contains("Block 1"));
        assert!(svg.contains("Block 2"));
        assert!(svg.contains("Block 3"));
        assert!(svg.contains("Block 4"));

        // Should have arrows (path elements)
        assert!(svg.contains("<path"));
    }
}
