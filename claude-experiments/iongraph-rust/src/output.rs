use crate::types::*;
use crate::graph::Graph;
use serde_json;
use std::io::{self, Write};
use std::fs;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayoutOutput {
    pub metadata: LayoutMetadata,
    pub blocks: Vec<LayoutBlock>,
    pub edges: Vec<LayoutEdge>,
    pub layers: Vec<LayerInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayoutMetadata {
    pub graph_size: Vec2,
    pub viewport_size: Vec2,
    pub num_layers: usize,
    pub total_blocks: usize,
    pub layout_algorithm: String,
    pub constants: LayoutConstants,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayoutConstants {
    pub content_padding: f64,
    pub block_gap: f64,
    pub port_start: f64,
    pub port_spacing: f64,
    pub arrow_radius: f64,
    pub joint_spacing: f64,
    pub track_padding: f64,
}

impl Default for LayoutConstants {
    fn default() -> Self {
        Self {
            content_padding: CONTENT_PADDING,
            block_gap: BLOCK_GAP,
            port_start: PORT_START,
            port_spacing: PORT_SPACING,
            arrow_radius: ARROW_RADIUS,
            joint_spacing: JOINT_SPACING,
            track_padding: TRACK_PADDING,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayoutBlock {
    pub id: u32,
    pub number: u32,
    pub position: Vec2,
    pub size: Vec2,
    pub layer: usize,
    pub attributes: Vec<String>,
    pub loop_depth: u32,
    pub successors: Vec<u32>,
    pub predecessors: Vec<u32>,
    pub instruction_count: u32,
    pub is_dummy: bool,
    pub node_flags: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayoutEdge {
    pub source_block: u32,
    pub target_block: u32,
    pub source_port: usize,
    pub joint_offset: f64,
    pub is_backedge: bool,
    pub path_type: EdgePathType,
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum EdgePathType {
    Straight,
    Curved,
    Backedge,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerInfo {
    pub index: usize,
    pub height: f64,
    pub track_height: f64,
    pub y_position: f64,
    pub block_count: usize,
    pub dummy_count: usize,
}

impl LayoutOutput {
    pub fn from_graph(graph: &mut Graph) -> Self {
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        
        // Calculate graph size
        let mut max_x = 0.0;
        let mut max_y = 0.0;
        for layer in &nodes_by_layer {
            for node in layer {
                max_x = max_x.max(node.pos.x + node.size.x);
                max_y = max_y.max(node.pos.y + node.size.y);
            }
        }
        let graph_size = Vec2::new(max_x + CONTENT_PADDING, max_y + CONTENT_PADDING);

        // Build metadata
        let metadata = LayoutMetadata {
            graph_size,
            viewport_size: graph.viewport_size.clone(),
            num_layers: nodes_by_layer.len(),
            total_blocks: graph.blocks.len(),
            layout_algorithm: "iongraph-rust".to_string(),
            constants: LayoutConstants::default(),
        };

        // Convert layout nodes to output format
        let mut blocks = Vec::new();
        let mut edges = Vec::new();
        let mut node_lookup = std::collections::HashMap::new();

        // Build node lookup for edge creation
        for layer in &nodes_by_layer {
            for node in layer {
                node_lookup.insert(node.id, node);
            }
        }

        // Process nodes and edges
        for layer in &nodes_by_layer {
            for node in layer {
                // Create block entry
                let layout_block = if let Some(block) = &node.block {
                    LayoutBlock {
                        id: node.id,
                        number: block.number.0,
                        position: node.pos.clone(),
                        size: node.size.clone(),
                        layer: node.layer,
                        attributes: block.attributes.clone(),
                        loop_depth: block.loop_depth,
                        successors: block.successors.iter().map(|s| s.0).collect(),
                        predecessors: block.predecessors.iter().map(|p| p.0).collect(),
                        instruction_count: block.instruction_count,
                        is_dummy: false,
                        node_flags: node.flags,
                    }
                } else if let Some(dst_block) = &node.dst_block {
                    LayoutBlock {
                        id: node.id,
                        number: dst_block.number.0,
                        position: node.pos.clone(),
                        size: node.size.clone(),
                        layer: node.layer,
                        attributes: vec!["dummy".to_string()],
                        loop_depth: dst_block.loop_depth,
                        successors: vec![dst_block.number.0],
                        predecessors: vec![],
                        instruction_count: 0,
                        is_dummy: true,
                        node_flags: node.flags,
                    }
                } else {
                    continue; // Skip malformed nodes
                };
                
                blocks.push(layout_block);

                // Create edges for this node's destinations
                for (port_idx, &dst_id) in node.dst_nodes.iter().enumerate() {
                    if let Some(dst_node) = node_lookup.get(&dst_id) {
                        let joint_offset = node.joint_offsets.get(port_idx).copied().unwrap_or(0.0);
                        let is_backedge = node.layer > dst_node.layer;
                        
                        let path_type = if is_backedge {
                            EdgePathType::Backedge
                        } else if joint_offset != 0.0 || (node.pos.x - dst_node.pos.x).abs() > 2.0 * ARROW_RADIUS {
                            EdgePathType::Curved  
                        } else {
                            EdgePathType::Straight
                        };

                        let target_block = if let Some(block) = &dst_node.block {
                            block.number.0
                        } else if let Some(dst_block) = &dst_node.dst_block {
                            dst_block.number.0
                        } else {
                            continue;
                        };

                        let source_block = if let Some(block) = &node.block {
                            block.number.0
                        } else if let Some(dst_block) = &node.dst_block {
                            dst_block.number.0
                        } else {
                            continue;
                        };

                        edges.push(LayoutEdge {
                            source_block,
                            target_block,
                            source_port: port_idx,
                            joint_offset,
                            is_backedge,
                            path_type,
                        });
                    }
                }
            }
        }

        // Build layer information
        let mut layers = Vec::new();
        let mut y_pos = CONTENT_PADDING;
        
        for (i, layer) in nodes_by_layer.iter().enumerate() {
            let layer_height = layer_heights.get(i).copied().unwrap_or(0.0);
            let track_height = track_heights.get(i).copied().unwrap_or(0.0);
            
            let block_count = layer.iter().filter(|n| !n.is_dummy()).count();
            let dummy_count = layer.iter().filter(|n| n.is_dummy()).count();
            
            layers.push(LayerInfo {
                index: i,
                height: layer_height,
                track_height,
                y_position: y_pos,
                block_count,
                dummy_count,
            });
            
            y_pos += layer_height + TRACK_PADDING + track_height + TRACK_PADDING;
        }

        LayoutOutput {
            metadata,
            blocks,
            edges,
            layers,
        }
    }

    pub fn to_json(&self) -> Result<String, OutputError> {
        serde_json::to_string_pretty(self).map_err(OutputError::Json)
    }

    pub fn to_compact_json(&self) -> Result<String, OutputError> {
        serde_json::to_string(self).map_err(OutputError::Json)
    }

    pub fn write_to_file(&self, path: &str) -> Result<(), OutputError> {
        let json = self.to_json()?;
        fs::write(path, json).map_err(OutputError::Io)
    }

    pub fn write_to_stdout(&self) -> Result<(), OutputError> {
        let json = self.to_json()?;
        print!("{}", json);
        io::stdout().flush().map_err(OutputError::Io)
    }
}

pub fn write_svg_to_file(svg: &str, path: &str) -> Result<(), OutputError> {
    fs::write(path, svg).map_err(OutputError::Io)
}

pub fn write_svg_to_stdout(svg: &str) -> Result<(), OutputError> {
    print!("{}", svg);
    io::stdout().flush().map_err(OutputError::Io)
}

#[derive(Debug)]
pub enum OutputError {
    Json(serde_json::Error),
    Io(io::Error),
}

impl std::fmt::Display for OutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputError::Json(e) => write!(f, "JSON serialization error: {}", e),
            OutputError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for OutputError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::*;

    #[test]
    fn test_layout_output_creation() {
        let pass = create_simple_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        
        let output = LayoutOutput::from_graph(&mut graph);
        
        assert_eq!(output.blocks.len(), 2);
        assert!(!output.edges.is_empty());
        assert!(!output.layers.is_empty());
        assert_eq!(output.metadata.total_blocks, 2);
    }

    #[test]
    fn test_json_serialization() {
        let pass = create_simple_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        
        let output = LayoutOutput::from_graph(&mut graph);
        let json = output.to_json().unwrap();
        
        assert!(json.contains("metadata"));
        assert!(json.contains("blocks"));
        assert!(json.contains("edges"));
        assert!(json.contains("layers"));
    }

    #[test]
    fn test_complex_layout_output() {
        let pass = create_complex_pass();
        let mut graph = Graph::new(Vec2::new(800.0, 600.0), pass);
        
        let output = LayoutOutput::from_graph(&mut graph);
        
        assert_eq!(output.blocks.len(), 5); // 5 blocks in complex pass
        assert!(output.edges.len() >= 4); // At least 4 edges connecting blocks
        assert!(output.layers.len() >= 3); // Multiple layers
    }
}