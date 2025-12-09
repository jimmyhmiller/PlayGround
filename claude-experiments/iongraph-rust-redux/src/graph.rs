// Port of Graph.ts - Core graph structure and rendering

// No longer need Ion-specific types
use crate::compilers::universal::{UniversalIR, UniversalInstruction};
use crate::layout_provider::{LayoutProvider, Vec2};
use std::collections::HashSet;

pub const CONTENT_PADDING: f64 = 20.0;
pub const BLOCK_GAP: f64 = 44.0;
pub const PORT_START: f64 = 16.0;
pub const PORT_SPACING: f64 = 60.0;
pub const ARROW_RADIUS: f64 = 12.0;
pub const TRACK_PADDING: f64 = 36.0;
pub const JOINT_SPACING: f64 = 16.0;
pub const HEADER_ARROW_PUSHDOWN: f64 = 16.0;

pub const LAYOUT_ITERATIONS: i32 = 2;
pub const NEARLY_STRAIGHT: f64 = 30.0;
pub const NEARLY_STRAIGHT_ITERATIONS: i32 = 8;
pub const STOP_AT_PASS: i32 = 30;

/// Sentinel value for uninitialized dst_nodes entries (must match graph_layout.rs)
const NO_DESTINATION: usize = usize::MAX;

// Constants for future interactivity features
#[allow(dead_code)]
const ZOOM_SENSITIVITY: f64 = 1.50;
#[allow(dead_code)]
const WHEEL_DELTA_SCALE: f64 = 0.01;
#[allow(dead_code)]
const MAX_ZOOM: f64 = 1.0;
#[allow(dead_code)]
const MIN_ZOOM: f64 = 0.10;
#[allow(dead_code)]
const TRANSLATION_CLAMP_AMOUNT: f64 = 40.0;

pub struct Block<E> {
    pub id: String,  // Universal format uses string IDs
    pub attributes: Vec<String>,
    pub predecessors: Vec<usize>, // indices into blocks vec
    pub successors: Vec<usize>,   // indices into blocks vec
    pub back_edges: Vec<usize>,   // indices into blocks vec (for rendering only)
    pub has_self_loop: bool,      // true if this block has a self-loop edge
    pub succs: Vec<usize>,        // Copy for convenience
    pub instructions: Vec<UniversalInstruction>,
    pub element: Option<Box<E>>,
    pub size: Vec2,
    pub layer: i32,
    pub loop_id: String,  // String ID instead of BlockID
    pub loop_depth: u32,
    pub layout_node: Option<usize>, // Index into layout nodes (global)
}

impl<E> Block<E> {
    /// Check if this block has a semantic attribute
    pub fn has_semantic_attribute(&self, semantic: crate::core::SemanticAttribute) -> bool {
        use crate::core::semantic_attrs::AttributeSemantics;
        use crate::compilers::UniversalCompilerIR;

        UniversalCompilerIR::has_semantic_attribute(&self.attributes, semantic)
    }

    /// Check if this block is a loop header
    pub fn is_loop_header(&self) -> bool {
        self.has_semantic_attribute(crate::core::SemanticAttribute::LoopHeader)
    }

    /// Check if this block is a backedge
    pub fn is_backedge(&self) -> bool {
        self.has_semantic_attribute(crate::core::SemanticAttribute::Backedge)
    }
}

#[derive(Clone)]
pub struct SampleCounts {
    // Placeholder for sample count data
}

pub struct LoopHeader {
    pub block_idx: usize,
    pub loop_height: f64,
    pub parent_loop: Option<usize>,
    pub outgoing_edges: Vec<usize>,
    pub backedge: usize,
}

#[derive(Clone)]
pub struct BlockNode {
    pub id: usize,
    pub pos: Vec2,
    pub size: Vec2,
    pub src_nodes: Vec<usize>, // global indices
    pub dst_nodes: Vec<usize>, // global indices
    pub joint_offsets: Vec<f64>,
    pub flags: NodeFlags,
    pub block: usize, // index into blocks vec
}

#[derive(Clone)]
pub struct DummyNode {
    pub id: usize,
    pub pos: Vec2,
    pub size: Vec2,
    pub src_nodes: Vec<usize>, // indices
    pub dst_nodes: Vec<usize>, // indices
    pub joint_offsets: Vec<f64>,
    pub flags: NodeFlags,
    pub dst_block: usize, // index into blocks vec
}

pub type LayoutNodeID = usize;
pub type NodeFlags = u32;

pub const LEFTMOST_DUMMY: NodeFlags = 1 << 0;
pub const RIGHTMOST_DUMMY: NodeFlags = 1 << 1;
pub const IMMINENT_BACKEDGE_DUMMY: NodeFlags = 1 << 2;

pub const SC_TOTAL: usize = 0;
pub const SC_SELF: usize = 1;

pub struct GraphNavigation {
    pub visited: Vec<String>,  // Block IDs
    pub current_index: i32,
    pub siblings: Vec<String>,  // Block IDs
}

pub struct HighlightedInstruction {
    pub id: String,  // Instruction ID from metadata
    pub palette_color: usize,
}

pub struct GraphState {
    pub translation: Vec2,
    pub zoom: f64,
    pub heatmap_mode: usize,
    pub highlighted_instructions: Vec<HighlightedInstruction>,
    pub selected_block_ids: HashSet<String>,  // Block IDs
    pub last_selected_block_id: String,  // Block ID
    pub viewport_pos_of_selected_block: Option<Vec2>,
}

pub struct RestoreStateOpts {
    pub preserve_selected_block_position: bool,
}

#[derive(Clone)]
pub struct GraphOptions {
    pub sample_counts: Option<SampleCounts>,
    pub instruction_palette: Option<Vec<String>>,
}

pub struct Graph<P: LayoutProvider> {
    // Layout provider
    pub layout_provider: P,

    // HTML elements
    pub viewport: Box<P::Element>,
    pub viewport_size: Vec2,
    pub graph_container: Box<P::Element>,

    // Core graph data (now Universal IR format)
    pub ir: UniversalIR,
    pub blocks: Vec<Block<P::Element>>,
    pub blocks_by_id: std::collections::HashMap<String, usize>,  // String IDs now

    // Layout state
    pub loops: Vec<LoopHeader>,
    pub num_layers: usize,

    // Post-layout info
    pub size: Vec2,

    // Options
    pub sample_counts: Option<SampleCounts>,
    pub instruction_palette: Option<Vec<String>>,

    // Navigation state
    pub navigation: GraphNavigation,
    pub state: GraphState,
}

#[derive(Clone)]
pub enum LayoutNode {
    BlockNode(BlockNode),
    DummyNode(DummyNode),
}

impl<P: LayoutProvider> Graph<P> {
    pub fn new(mut layout_provider: P, ir: UniversalIR, options: GraphOptions) -> Self {
        let viewport = layout_provider.create_element("div");
        let mut graph_container = layout_provider.create_svg_element("g");
        layout_provider.add_class(&mut graph_container, "ig-graph");

        let mut graph = Graph {
            layout_provider,
            viewport,
            viewport_size: Vec2 { x: 800.0, y: 600.0 },
            graph_container,
            ir,
            blocks: Vec::new(),
            blocks_by_id: std::collections::HashMap::new(),
            loops: Vec::new(),
            num_layers: 0,
            size: Vec2 { x: 0.0, y: 0.0 },
            sample_counts: options.sample_counts,
            instruction_palette: options.instruction_palette,
            navigation: GraphNavigation {
                visited: Vec::new(),
                current_index: -1,
                siblings: Vec::new(),
            },
            state: GraphState {
                translation: Vec2 { x: 0.0, y: 0.0 },
                zoom: 1.0,
                heatmap_mode: 0,
                highlighted_instructions: Vec::new(),
                selected_block_ids: HashSet::new(),
                last_selected_block_id: String::new(),
                viewport_pos_of_selected_block: None,
            },
        };

        graph.build_blocks();
        graph.render_blocks_for_measurement(); // Render blocks to DOM for size measurement
        graph.measure_block_sizes(); // Measure actual sizes from DOM
        graph
    }

    fn build_blocks(&mut self) {
        // Build blocks from Universal IR
        for universal_block in &self.ir.blocks {
            let block_idx = self.blocks.len();

            // Start with a placeholder size - will be measured after rendering
            let size = Vec2 { x: 0.0, y: 0.0 };

            let block = Block {
                id: universal_block.id.clone(),
                attributes: universal_block.attributes.clone(),
                predecessors: Vec::new(),
                successors: Vec::new(),
                back_edges: Vec::new(),
                has_self_loop: universal_block.has_self_loop,
                succs: Vec::new(),
                instructions: universal_block.instructions.clone(),
                element: None,
                size,
                layer: -1,
                loop_id: String::new(),
                loop_depth: universal_block.loop_depth,
                layout_node: None,
            };
            self.blocks.push(block);
            self.blocks_by_id.insert(universal_block.id.clone(), block_idx);
        }

        // Build successor/predecessor relationships
        for (idx, universal_block) in self.ir.blocks.iter().enumerate() {
            for succ_id in &universal_block.successors {
                if let Some(&succ_idx) = self.blocks_by_id.get(succ_id) {
                    self.blocks[idx].successors.push(succ_idx);
                    self.blocks[idx].succs.push(succ_idx);
                    self.blocks[succ_idx].predecessors.push(idx);
                }
            }
            // Build back_edges separately (for rendering only, not layout)
            for back_edge_id in &universal_block.back_edges {
                if let Some(&back_edge_idx) = self.blocks_by_id.get(back_edge_id) {
                    self.blocks[idx].back_edges.push(back_edge_idx);
                }
            }
        }
    }

    // Render all blocks to DOM so we can measure their actual sizes (TypeScript lines 318-319)
    fn render_blocks_for_measurement(&mut self) {
        for idx in 0..self.blocks.len() {
            let block_el = self.render_block_element(idx);
            self.blocks[idx].element = Some(block_el);
        }
    }

    // Measure actual sizes from rendered blocks (TypeScript lines 339-344)
    fn measure_block_sizes(&mut self) {
        for idx in 0..self.blocks.len() {
            if let Some(el) = &self.blocks[idx].element {
                let width = self.layout_provider.get_client_width(el);
                let height = self.layout_provider.get_client_height(el);

                // Clone el temporarily to avoid borrow checker issues
                if let Some(mut el) = self.blocks[idx].element.take() {
                    self.blocks[idx].size = Vec2 {
                        x: width,
                        y: height,
                    };

                    // Store sizes as attributes for SVG rendering
                    self.layout_provider
                        .set_attribute(&mut el, "data-width", &width.to_string());
                    self.layout_provider
                        .set_attribute(&mut el, "data-height", &height.to_string());

                    // Put el back
                    self.blocks[idx].element = Some(el);
                }
            }
        }
    }

    fn render_block_element(&mut self, block_idx: usize) -> Box<P::Element> {
        // Clone the data we need to avoid borrow checker issues
        let block_id = self.blocks[block_idx].id.clone();
        let attributes = self.blocks[block_idx].attributes.clone();
        let instructions = self.blocks[block_idx].instructions.clone();
        let num_successors = self.blocks[block_idx].successors.len();

        let mut el = self.layout_provider.create_element("div");

        self.layout_provider
            .add_classes(&mut el, &["ig-block", "ig-bg-white"]);

        for att in &attributes {
            let class_name = format!("ig-block-att-{}", att);
            self.layout_provider.add_class(&mut el, &class_name);
        }

        self.layout_provider
            .set_attribute(&mut el, "data-ig-block-id", &block_id);

        // Create header
        let mut desc = String::new();
        if attributes.contains(&"loopheader".to_string()) {
            desc = " (loop header)".to_string();
        } else if attributes.contains(&"backedge".to_string()) {
            desc = " (backedge)".to_string();
        } else if attributes.contains(&"splitedge".to_string()) {
            desc = " (split edge)".to_string();
        }

        let mut header = self.layout_provider.create_element("div");
        self.layout_provider
            .add_class(&mut header, "ig-block-header");
        self.layout_provider
            .set_inner_text(&mut header, &format!("Block {}{}", block_id, desc));
        self.layout_provider.append_child(&mut el, header);

        // Create instructions container
        let mut insns_container = self.layout_provider.create_element("div");
        self.layout_provider
            .add_class(&mut insns_container, "ig-instructions");

        let mut insns = self.layout_provider.create_element("table");

        for (idx, ins) in instructions.iter().enumerate() {
            let ins_row = self.render_universal_instruction(ins, idx);
            self.layout_provider.append_child(&mut insns, ins_row);
        }

        self.layout_provider
            .append_child(&mut insns_container, insns);
        self.layout_provider.append_child(&mut el, insns_container);

        // Add edge labels for binary branches
        if num_successors == 2 {
            for (i, label) in [1, 0].iter().enumerate() {
                let mut edge_label = self.layout_provider.create_element("div");
                self.layout_provider
                    .set_inner_text(&mut edge_label, &label.to_string());
                self.layout_provider
                    .add_class(&mut edge_label, "ig-edge-label");
                self.layout_provider.set_style(
                    &mut edge_label,
                    "left",
                    &format!("{}px", PORT_START + PORT_SPACING * i as f64),
                );
                self.layout_provider.append_child(&mut el, edge_label);
            }
        }

        el
    }

    fn render_universal_instruction(&mut self, ins: &UniversalInstruction, idx: usize) -> Box<P::Element> {
        let mut row = self.layout_provider.create_element("tr");
        self.layout_provider.add_class(&mut row, "ig-ins");

        // Add instruction attributes as classes
        for attr in &ins.attributes {
            let class_name = format!("ig-ins-att-{}", attr);
            self.layout_provider.add_class(&mut row, &class_name);
        }

        // ID column (use index if no ID in metadata)
        let mut id_cell = self.layout_provider.create_element("td");
        self.layout_provider.add_class(&mut id_cell, "ig-ins-num");
        let id_str = ins.metadata.get("id")
            .and_then(|v| v.as_u64())
            .map(|n| n.to_string())
            .unwrap_or_else(|| idx.to_string());
        self.layout_provider.set_inner_text(&mut id_cell, &id_str);
        self.layout_provider.append_child(&mut row, id_cell);

        // Opcode column
        let mut opcode_cell = self.layout_provider.create_element("td");
        self.layout_provider
            .set_inner_text(&mut opcode_cell, &ins.opcode);
        self.layout_provider.append_child(&mut row, opcode_cell);

        // Type column (if present)
        if let Some(ref type_) = ins.type_ {
            let mut type_cell = self.layout_provider.create_element("td");
            self.layout_provider
                .add_class(&mut type_cell, "ig-ins-type");
            self.layout_provider.set_inner_text(&mut type_cell, type_);
            self.layout_provider.append_child(&mut row, type_cell);
        }

        // Sample counts (if profiling data available)
        if let Some(ref profiling) = ins.profiling {
            // Total count column
            let mut total_cell = self.layout_provider.create_element("td");
            self.layout_provider
                .add_class(&mut total_cell, "ig-ins-samples");
            self.layout_provider.set_inner_text(&mut total_cell, &profiling.sample_count.to_string());
            self.layout_provider.append_child(&mut row, total_cell);
        }

        row
    }

    pub fn render(
        &mut self,
        nodes_by_layer: Vec<Vec<LayoutNode>>,
        layer_heights: Vec<f64>,
        track_heights: Vec<f64>,
    ) {
        // Add empty rect at the beginning (matches TypeScript)
        let empty_rect = self.layout_provider.create_svg_element("rect");
        self.layout_provider
            .append_child(&mut self.graph_container, empty_rect);

        // Add blocks to graph container in JSON order (matches TypeScript lines 318-319, 1228-1229)
        // TypeScript adds them in constructor, then positions them in render()
        // We do both here but maintain JSON order for SVG output
        for block_idx in 0..self.blocks.len() {
            if let Some(mut block_el) = self.blocks[block_idx].element.take() {
                // Position the block based on its layout_node position
                if let Some(_layout_node_idx) = self.blocks[block_idx].layout_node {
                    // Find this block's layout node to get its position
                    for layer in &nodes_by_layer {
                        for node in layer {
                            if let LayoutNode::BlockNode(block_node) = node {
                                if block_node.block == block_idx {
                                    // Position the block
                                    self.layout_provider.set_style(
                                        &mut block_el,
                                        "left",
                                        &format!("{}px", block_node.pos.x),
                                    );
                                    self.layout_provider.set_style(
                                        &mut block_el,
                                        "top",
                                        &format!("{}px", block_node.pos.y),
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }

                // Append to graph container (this consumes block_el)
                self.layout_provider
                    .append_child(&mut self.graph_container, block_el);

                // Element is now owned by graph_container, clear the reference
                self.blocks[block_idx].element = None;
            }
        }

        // Calculate total size
        let mut max_x: f64 = 0.0;
        let mut max_y: f64 = 0.0;

        for layer in nodes_by_layer.iter() {
            for node in layer {
                let (pos, size, _block_id, _flags) = match node {
                    LayoutNode::BlockNode(n) => {
                        (n.pos, n.size, Some(self.blocks[n.block].id.clone()), n.flags)
                    }
                    LayoutNode::DummyNode(n) => (n.pos, n.size, None, n.flags),
                };
                let candidate_y = pos.y + size.y + CONTENT_PADDING;
                if candidate_y > max_y {
                    max_y = candidate_y;
                }
                let candidate_x = pos.x + size.x + CONTENT_PADDING;
                if candidate_x > max_x {
                    max_x = candidate_x;
                }
            }
        }

        // Create container for arrows
        let mut arrows_container = self.layout_provider.create_svg_element("g");

        // Render arrows
        for (layer_idx, layer) in nodes_by_layer.iter().enumerate() {
            for node in layer {
                let (node_pos, node_size, node_flags, node_block_idx, dst_nodes, joint_offsets) =
                    match node {
                        LayoutNode::BlockNode(n) => (
                            n.pos,
                            n.size,
                            n.flags,
                            Some(n.block),
                            n.dst_nodes.clone(),
                            n.joint_offsets.clone(),
                        ),
                        LayoutNode::DummyNode(n) => (
                            n.pos,
                            n.size,
                            n.flags,
                            None,
                            n.dst_nodes.clone(),
                            n.joint_offsets.clone(),
                        ),
                    };

                // Iterate through destination nodes and draw arrows
                for (i, &dst_global_idx) in dst_nodes.iter().enumerate() {
                    // Skip uninitialized entries (padding from port assignment)
                    if dst_global_idx == NO_DESTINATION {
                        continue;
                    }

                    let x1 = node_pos.x + PORT_START + PORT_SPACING * i as f64;
                    let y1 = node_pos.y + node_size.y;

                    // Find the destination node in the layout
                    let dst_node =
                        self.find_layout_node_by_global_idx(&nodes_by_layer, dst_global_idx);

                    if let Some(dst) = dst_node {
                        let (dst_pos, _dst_size, dst_flags, _dst_block_idx) = match dst {
                            LayoutNode::BlockNode(n) => (n.pos, n.size, n.flags, Some(n.block)),
                            LayoutNode::DummyNode(n) => (n.pos, n.size, n.flags, Some(n.dst_block)),
                        };

                        // Check if this is a backedge block - draw loop header arrow
                        let is_backedge = if let Some(block_idx) = node_block_idx {
                            self.blocks[block_idx]
                                .attributes
                                .contains(&"backedge".to_string())
                        } else {
                            false
                        };

                        if is_backedge {
                            // Draw loop header arrow (TypeScript lines 1282-1290)
                            if let Some(block_idx) = node_block_idx {
                                if !self.blocks[block_idx].successors.is_empty() {
                                    let header_idx = self.blocks[block_idx].successors[0];
                                    if let Some(header_layout_node) =
                                        self.blocks[header_idx].layout_node
                                    {
                                        let header_node = self.find_layout_node_by_global_idx(
                                            &nodes_by_layer,
                                            header_layout_node,
                                        );
                                        if let Some(LayoutNode::BlockNode(header)) = header_node {
                                            let x1_arrow = node_pos.x;
                                            let y1_arrow = node_pos.y + HEADER_ARROW_PUSHDOWN;
                                            let x2_arrow = header.pos.x + header.size.x;
                                            let y2_arrow = header.pos.y + HEADER_ARROW_PUSHDOWN;
                                            let arrow = loop_header_arrow(
                                                &mut self.layout_provider,
                                                x1_arrow,
                                                y1_arrow,
                                                x2_arrow,
                                                y2_arrow,
                                                1,
                                            );
                                            self.layout_provider
                                                .append_child(&mut arrows_container, arrow);
                                        }
                                    }
                                }
                            }
                        } else if (node_flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
                            // Draw from IMMINENT_BACKEDGE_DUMMY to backedge (TypeScript lines 1291-1299)
                            if let LayoutNode::DummyNode(dummy) = node {
                                let backedge_idx = dummy.dst_block;
                                if let Some(backedge_layout_node) =
                                    self.blocks[backedge_idx].layout_node
                                {
                                    let backedge_node = self.find_layout_node_by_global_idx(
                                        &nodes_by_layer,
                                        backedge_layout_node,
                                    );
                                    if let Some(LayoutNode::BlockNode(backedge)) = backedge_node {
                                        let x1_arrow = node_pos.x + PORT_START;
                                        let y1_arrow =
                                            node_pos.y + HEADER_ARROW_PUSHDOWN + ARROW_RADIUS;
                                        let x2_arrow = backedge.pos.x + backedge.size.x;
                                        let y2_arrow = backedge.pos.y + HEADER_ARROW_PUSHDOWN;
                                        let arrow = arrow_to_backedge(
                                            &mut self.layout_provider,
                                            x1_arrow,
                                            y1_arrow,
                                            x2_arrow,
                                            y2_arrow,
                                            1,
                                        );
                                        self.layout_provider
                                            .append_child(&mut arrows_container, arrow);
                                    }
                                }
                            }
                        } else if matches!(dst, LayoutNode::DummyNode(_)) {
                            // Check if this dummy eventually leads to a backedge (TypeScript line 1279)
                            let dst_dummy_block = if let LayoutNode::DummyNode(dn) = dst {
                                dn.dst_block
                            } else {
                                unreachable!()
                            };
                            let dst_dummy_leads_to_backedge = self.blocks[dst_dummy_block]
                                .attributes
                                .contains(&"backedge".to_string());

                            if dst_dummy_leads_to_backedge {
                                // Arrow to backedge dummy (TypeScript lines 1279-1292)
                                let x2 = dst_pos.x + PORT_START;
                                let y2 = dst_pos.y
                                    + if (dst_flags & IMMINENT_BACKEDGE_DUMMY) != 0 {
                                        HEADER_ARROW_PUSHDOWN + ARROW_RADIUS
                                    } else {
                                        0.0
                                    };

                                if node_block_idx.is_none() {
                                    // Draw upward arrow between dummies
                                    let ym = y1 - TRACK_PADDING;
                                    let arrow = upward_arrow(
                                        &mut self.layout_provider,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        ym,
                                        false,
                                        1,
                                    );
                                    self.layout_provider
                                        .append_child(&mut arrows_container, arrow);
                                } else {
                                    // Draw arrow from block to backedge dummy
                                    let ym = (y1 - node_size.y)
                                        + layer_heights[layer_idx]
                                        + TRACK_PADDING
                                        + track_heights[layer_idx] / 2.0
                                        + joint_offsets[i];
                                    let arrow = arrow_from_block_to_backedge_dummy(
                                        &mut self.layout_provider,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        ym,
                                        1,
                                    );
                                    self.layout_provider
                                        .append_child(&mut arrows_container, arrow);
                                }
                            } else {
                                // Regular downward arrow to dummy
                                let x2 = dst_pos.x + PORT_START;
                                let y2 = dst_pos.y;
                                let ym = (y1 - node_size.y)
                                    + layer_heights[layer_idx]
                                    + TRACK_PADDING
                                    + track_heights[layer_idx] / 2.0
                                    + joint_offsets[i];
                                let arrow = downward_arrow(
                                    &mut self.layout_provider,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    ym,
                                    false,
                                    1,
                                );
                                self.layout_provider
                                    .append_child(&mut arrows_container, arrow);
                            }
                        } else {
                            let x2 = dst_pos.x + PORT_START;
                            let y2 = dst_pos.y;
                            let do_arrowhead = matches!(dst, LayoutNode::BlockNode(_));

                            // Check if this is a back edge (destination is above source)
                            if y2 < y1 {
                                // Back edge: draw curved upward arrow to the right side
                                // The arrow goes out to the right, up, and curves back to the destination
                                let arrow = back_edge_arrow(
                                    &mut self.layout_provider,
                                    x1,
                                    y1,
                                    x2,
                                    y2, // Target the top edge of the block
                                    do_arrowhead,
                                    1,
                                );
                                self.layout_provider
                                    .append_child(&mut arrows_container, arrow);
                            } else {
                                // Regular downward arrow
                                let ym = (y1 - node_size.y)
                                    + layer_heights[layer_idx]
                                    + TRACK_PADDING
                                    + track_heights[layer_idx] / 2.0
                                    + joint_offsets[i];
                                let arrow = downward_arrow(
                                    &mut self.layout_provider,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    ym,
                                    do_arrowhead,
                                    1,
                                );
                                self.layout_provider
                                    .append_child(&mut arrows_container, arrow);
                            }
                        }
                    }
                }
            }
        }

        // Render back edges (bidirectional edge support)
        // Back edges are stored separately and rendered as curved arrows going up and around
        for block_idx in 0..self.blocks.len() {
            if self.blocks[block_idx].back_edges.is_empty() {
                continue;
            }

            // Find the source block's layout node
            let source_layout_idx = match self.blocks[block_idx].layout_node {
                Some(idx) => idx,
                None => continue,
            };

            let source_node = self.find_layout_node_by_global_idx(&nodes_by_layer, source_layout_idx);
            let (source_pos, source_size) = match source_node {
                Some(LayoutNode::BlockNode(n)) => (n.pos, n.size),
                _ => continue,
            };

            // Render each back edge
            for (i, &target_idx) in self.blocks[block_idx].back_edges.iter().enumerate() {
                let target_layout_idx = match self.blocks[target_idx].layout_node {
                    Some(idx) => idx,
                    None => continue,
                };

                let target_node = self.find_layout_node_by_global_idx(&nodes_by_layer, target_layout_idx);
                let (target_pos, _target_size) = match target_node {
                    Some(LayoutNode::BlockNode(n)) => (n.pos, n.size),
                    _ => continue,
                };

                // Calculate arrow start (bottom of source block, offset by port)
                let num_successors = self.blocks[block_idx].successors.len();
                let x1 = source_pos.x + PORT_START + PORT_SPACING * (num_successors + i) as f64;
                let y1 = source_pos.y + source_size.y;

                // Calculate arrow end (top edge of target block)
                let x2 = target_pos.x + PORT_START;
                let y2 = target_pos.y; // Top edge, not inside header

                let arrow = back_edge_arrow(
                    &mut self.layout_provider,
                    x1,
                    y1,
                    x2,
                    y2,
                    true, // do_arrowhead
                    1,    // stroke
                );
                self.layout_provider.append_child(&mut arrows_container, arrow);
            }
        }

        // Render self-loops
        for block_idx in 0..self.blocks.len() {
            if !self.blocks[block_idx].has_self_loop {
                continue;
            }

            // Find the block's layout node
            let layout_idx = match self.blocks[block_idx].layout_node {
                Some(idx) => idx,
                None => continue,
            };

            let block_node = self.find_layout_node_by_global_idx(&nodes_by_layer, layout_idx);
            let (block_pos, block_size) = match block_node {
                Some(LayoutNode::BlockNode(n)) => (n.pos, n.size),
                _ => continue,
            };

            // Draw self-loop arrow (starts from right side middle, curves out and back to header)
            let arrow = self_loop_arrow(
                &mut self.layout_provider,
                block_pos.x,
                block_pos.y,
                block_size.x,
                block_size.y,
                1,
            );
            self.layout_provider.append_child(&mut arrows_container, arrow);
        }

        // Append arrows container to graph
        self.layout_provider
            .append_child(&mut self.graph_container, arrows_container);

        // Store graph size
        self.size = Vec2 { x: max_x, y: max_y };
    }

    fn find_layout_node_by_global_idx<'a>(
        &self,
        nodes_by_layer: &'a [Vec<LayoutNode>],
        global_idx: usize,
    ) -> Option<&'a LayoutNode> {
        for layer in nodes_by_layer {
            for node in layer {
                let node_id = match node {
                    LayoutNode::BlockNode(n) => n.id,
                    LayoutNode::DummyNode(n) => n.id,
                };
                if node_id == global_idx {
                    return Some(node);
                }
            }
        }
        None
    }
}

// Arrow rendering functions
#[allow(clippy::too_many_arguments)]
fn downward_arrow<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ym: f64,
    do_arrowhead: bool,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;
    // In production, we'd assert here: assert!(y1 + r <= ym && ym < y2 - r)

    // Align stroke to pixels
    let mut x1 = x1;
    let mut x2 = x2;
    let mut ym = ym;
    if stroke % 2 == 1 {
        x1 += 0.5;
        x2 += 0.5;
        ym += 0.5;
    }

    let mut path = String::new();
    path.push_str(&format!("M {} {} ", x1, y1)); // move to start

    if (x2 - x1).abs() < 2.0 * r {
        // Degenerate case where the radii won't fit; fall back to bezier.
        path.push_str(&format!(
            "C {} {} {} {} {} {} ",
            x1,
            y1 + (y2 - y1) / 3.0,
            x2,
            y1 + 2.0 * (y2 - y1) / 3.0,
            x2,
            y2
        ));
    } else {
        let dir = (x2 - x1).signum();
        path.push_str(&format!("L {} {} ", x1, ym - r)); // line down
        path.push_str(&format!(
            "A {} {} 0 0 {} {} {} ",
            r,
            r,
            if dir > 0.0 { 0 } else { 1 },
            x1 + r * dir,
            ym
        )); // arc to joint
        path.push_str(&format!("L {} {} ", x2 - r * dir, ym)); // joint
        path.push_str(&format!(
            "A {} {} 0 0 {} {} {} ",
            r,
            r,
            if dir > 0.0 { 1 } else { 0 },
            x2,
            ym + r
        )); // arc to line
        path.push_str(&format!("L {} {} ", x2, y2)); // line down
    }

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke)); // Add trailing space to match TypeScript
    layout_provider.append_child(&mut g, p);

    if do_arrowhead {
        let v = arrowhead(layout_provider, x2, y2, 180, 5);
        layout_provider.append_child(&mut g, v);
    }

    g
}

#[allow(clippy::too_many_arguments)]
fn upward_arrow<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ym: f64,
    do_arrowhead: bool,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;
    // In production, we'd assert here: assert!(y2 + r <= ym && ym <= y1 - r)

    // Align stroke to pixels
    let mut x1 = x1;
    let mut x2 = x2;
    let mut ym = ym;
    if stroke % 2 == 1 {
        x1 += 0.5;
        x2 += 0.5;
        ym += 0.5;
    }

    let mut path = String::new();
    path.push_str(&format!("M {} {} ", x1, y1)); // move to start

    if (x2 - x1).abs() < 2.0 * r {
        // Degenerate case where the radii won't fit; fall back to bezier.
        path.push_str(&format!(
            "C {} {} {} {} {} {} ",
            x1,
            y1 + (y2 - y1) / 3.0,
            x2,
            y1 + 2.0 * (y2 - y1) / 3.0,
            x2,
            y2
        ));
    } else {
        let dir = (x2 - x1).signum();
        path.push_str(&format!("L {} {} ", x1, ym + r)); // line up
        path.push_str(&format!(
            "A {} {} 0 0 {} {} {} ",
            r,
            r,
            if dir > 0.0 { 1 } else { 0 },
            x1 + r * dir,
            ym
        )); // arc to joint
        path.push_str(&format!("L {} {} ", x2 - r * dir, ym)); // joint
        path.push_str(&format!(
            "A {} {} 0 0 {} {} {} ",
            r,
            r,
            if dir > 0.0 { 0 } else { 1 },
            x2,
            ym - r
        )); // arc to line
        path.push_str(&format!("L {} {} ", x2, y2)); // line up
    }

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke)); // Add trailing space to match TypeScript
    layout_provider.append_child(&mut g, p);

    if do_arrowhead {
        let v = arrowhead(layout_provider, x2, y2, 0, 5);
        layout_provider.append_child(&mut g, v);
    }

    g
}

fn arrow_to_backedge<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;
    // In production, we'd assert here: assert!(y1 - r >= y2 && x1 - r >= x2)

    // Align stroke to pixels
    let mut x1 = x1;
    let mut y2 = y2;
    if stroke % 2 == 1 {
        x1 += 0.5;
        y2 += 0.5;
    }

    let mut path = String::new();
    path.push_str(&format!("M {} {} ", x1, y1)); // move to start
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x1 - r, y2)); // arc to line
    path.push_str(&format!("L {} {} ", x2, y2)); // line left

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke)); // Add trailing space to match TypeScript
    layout_provider.append_child(&mut g, p);

    let v = arrowhead(layout_provider, x2, y2, 270, 5);
    layout_provider.append_child(&mut g, v);

    g
}

fn arrow_from_block_to_backedge_dummy<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ym: f64,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;
    // In production, we'd assert here: assert!(y1 + r <= ym && x1 <= x2 && y2 <= y1)

    // Align stroke to pixels
    let mut x1 = x1;
    let mut x2 = x2;
    let mut ym = ym;
    if stroke % 2 == 1 {
        x1 += 0.5;
        x2 += 0.5;
        ym += 0.5;
    }

    let mut path = String::new();
    path.push_str(&format!("M {} {} ", x1, y1)); // move to start
    path.push_str(&format!("L {} {} ", x1, ym - r)); // line down
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x1 + r, ym)); // arc to horizontal joint
    path.push_str(&format!("L {} {} ", x2 - r, ym)); // horizontal joint
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x2, ym - r)); // arc to line
    path.push_str(&format!("L {} {} ", x2, y2)); // line up

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke)); // Add trailing space to match TypeScript
    layout_provider.append_child(&mut g, p);

    g
}

fn loop_header_arrow<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: i32,
) -> Box<P::Element> {
    // In production, we'd assert here: assert!(x2 < x1 && y2 == y1)

    // Align stroke to pixels
    let mut y1 = y1;
    let mut y2 = y2;
    if stroke % 2 == 1 {
        y1 += 0.5;
        y2 += 0.5;
    }

    let mut path = String::new();
    path.push_str(&format!("M {} {} ", x1, y1)); // move to start
    path.push_str(&format!("L {} {} ", x2, y2)); // line left

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke)); // Add trailing space to match TypeScript
    layout_provider.append_child(&mut g, p);

    let v = arrowhead(layout_provider, x2, y2, 270, 5);
    layout_provider.append_child(&mut g, v);

    g
}

/// Draw a back edge arrow using elbow-style routing (orthogonal lines with arc corners)
/// Used for bidirectional edges where the destination is above the source
/// Routes: right from source, up past destination, left, then down into destination header
#[allow(clippy::too_many_arguments)]
fn back_edge_arrow<P: LayoutProvider>(
    layout_provider: &mut P,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    do_arrowhead: bool,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;

    // Align stroke to pixels
    let mut x1 = x1;
    let mut x2 = x2;
    if stroke % 2 == 1 {
        x1 += 0.5;
        x2 += 0.5;
    }

    // Calculate how far right to extend (based on the vertical distance)
    let vertical_distance = (y1 - y2).abs();
    let horizontal_offset = (vertical_distance / 3.0).max(40.0).min(80.0);
    let x_right = x1.max(x2) + horizontal_offset;

    // Offset the destination x to the right (don't land on left edge of block)
    let x_dest = x2 + horizontal_offset;

    // Y coordinate above the destination header
    let y_above = y2 - 1.5 * r;

    // Elbow path: right, up past destination, left, then down into header
    // Visual:
    //              ↓
    //          ┌───┘
    //          │
    //   Start ─┘
    let mut path = String::new();

    // Start at source (bottom of source block)
    path.push_str(&format!("M {} {} ", x1, y1));

    // Horizontal line right to just before the corner
    path.push_str(&format!("L {} {} ", x_right - r, y1));

    // Arc: bottom-right corner (right→up) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_right, y1 - r));

    // Vertical line going up past the destination
    path.push_str(&format!("L {} {} ", x_right, y_above + r));

    // Arc: top-right corner (up→left) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_right - r, y_above));

    // Horizontal line left (above the destination header)
    path.push_str(&format!("L {} {} ", x_dest + r, y_above));

    // Arc: top-left corner (left→down) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_dest, y_above + r));

    // Vertical line down to destination header
    path.push_str(&format!("L {} {} ", x_dest, y2));

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke));
    layout_provider.append_child(&mut g, p);

    if do_arrowhead {
        // Arrowhead pointing down into the destination header
        let v = arrowhead(layout_provider, x_dest, y2, 180, 5);
        layout_provider.append_child(&mut g, v);
    }

    g
}

/// Draw a self-loop arrow using elbow-style routing (orthogonal lines with arc corners)
/// Routes: right from middle of block, up, then left back to block header
#[allow(clippy::too_many_arguments)]
fn self_loop_arrow<P: LayoutProvider>(
    layout_provider: &mut P,
    block_x: f64,
    block_y: f64,
    block_width: f64,
    block_height: f64,
    stroke: i32,
) -> Box<P::Element> {
    let r = ARROW_RADIUS;
    let loop_extend = 35.0; // How far right of the block edge the loop extends

    // Start point: middle-right of block (on the edge)
    let x_start = block_x + block_width;
    let y_start = block_y + block_height / 2.0;

    // End point: offset from right edge so arrow comes down into header
    let x_end = block_x + block_width - loop_extend;
    let y_end = block_y;

    // Rightmost point of the loop
    let x_right = x_start + loop_extend;

    // Align to pixels for crisp rendering
    let x_start = if stroke % 2 == 1 { x_start + 0.5 } else { x_start };
    let x_end = if stroke % 2 == 1 { x_end + 0.5 } else { x_end };
    let x_right = if stroke % 2 == 1 { x_right + 0.5 } else { x_right };

    // Elbow path: right, up, left (above header), then down into header
    // Visual:
    //        ↓
    //    ┌───┘
    //    │
    //    │
    //    └───── ← start (middle right of block)
    let mut path = String::new();

    // Y coordinate for the horizontal segment above the header
    // Use 1.5*r to go higher and leave room for a straight vertical segment before the arrowhead
    let y_above = y_end - 1.5 * r;

    // Start at middle-right of block
    path.push_str(&format!("M {} {} ", x_start, y_start));

    // Horizontal line right to just before the corner
    path.push_str(&format!("L {} {} ", x_right - r, y_start));

    // Arc: bottom-right corner (right→up) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_right, y_start - r));

    // Vertical line going up
    path.push_str(&format!("L {} {} ", x_right, y_above + r));

    // Arc: top-right corner (up→left) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_right - r, y_above));

    // Horizontal line left (above the header)
    path.push_str(&format!("L {} {} ", x_end + r, y_above));

    // Arc: top-left corner (left→down) - sweep=0 for outward bulge
    path.push_str(&format!("A {} {} 0 0 0 {} {} ", r, r, x_end, y_above + r));

    // Vertical line down to header
    path.push_str(&format!("L {} {} ", x_end, y_end));

    let mut g = layout_provider.create_svg_element("g");

    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(&mut p, "d", &path);
    layout_provider.set_attribute(&mut p, "fill", "none");
    layout_provider.set_attribute(&mut p, "stroke", "black");
    layout_provider.set_attribute(&mut p, "stroke-width", &format!("{} ", stroke));
    layout_provider.append_child(&mut g, p);

    // Arrowhead pointing down into the block header
    let v = arrowhead(layout_provider, x_end, y_end, 180, 5);
    layout_provider.append_child(&mut g, v);

    g
}

fn arrowhead<P: LayoutProvider>(
    layout_provider: &mut P,
    x: f64,
    y: f64,
    rot: i32,
    size: i32,
) -> Box<P::Element> {
    let mut p = layout_provider.create_svg_element("path");
    layout_provider.set_attribute(
        &mut p,
        "d",
        &format!(
            "M 0 0 L {} {} L {} {} Z",
            -size,
            size as f64 * 1.5,
            size,
            size as f64 * 1.5
        ),
    );
    layout_provider.set_attribute(
        &mut p,
        "transform",
        &format!("translate({}, {}) rotate({})", x, y, rot),
    );
    p
}
