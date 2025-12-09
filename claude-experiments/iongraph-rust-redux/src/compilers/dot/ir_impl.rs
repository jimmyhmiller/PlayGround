// DOT file to UniversalIR conversion

use crate::compilers::universal::schema::{
    UniversalIR, UniversalBlock, UniversalInstruction, UNIVERSAL_VERSION,
};
use super::schema::{DotGraph, DotNode};
use std::collections::HashMap;

/// Detect cycles using DFS and find back edges
/// Returns:
/// - is_loop_header: which nodes are targets of back edges
/// - back_edges: set of (from, to) pairs that are back edges
fn detect_cycles(num_nodes: usize, edges: &[(usize, usize)]) -> (Vec<bool>, std::collections::HashSet<(usize, usize)>) {
    let mut is_loop_header = vec![false; num_nodes];
    let mut back_edges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    if num_nodes == 0 {
        return (is_loop_header, back_edges);
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for &(from, to) in edges {
        adj[from].push(to);
    }

    // DFS state
    let mut visited = vec![false; num_nodes];
    let mut in_stack = vec![false; num_nodes]; // Currently on recursion stack

    // Use iterative DFS to avoid stack overflow on large graphs
    for start in 0..num_nodes {
        if visited[start] {
            continue;
        }

        // Stack holds (node, iterator index into adjacency list)
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];

        while let Some(&(node, idx)) = stack.last() {
            if idx == 0 {
                // First visit to this node
                visited[node] = true;
                in_stack[node] = true;
            }

            if idx < adj[node].len() {
                let succ = adj[node][idx];
                // Move to next successor for when we return
                stack.last_mut().unwrap().1 += 1;

                if !visited[succ] {
                    stack.push((succ, 0));
                } else if in_stack[succ] {
                    // Back edge found: node → succ
                    back_edges.insert((node, succ));
                    is_loop_header[succ] = true;
                }
            } else {
                // Done with this node
                in_stack[node] = false;
                stack.pop();
            }
        }
    }

    (is_loop_header, back_edges)
}

/// Convert a DotGraph to UniversalIR format
pub fn dot_to_universal(graph: &DotGraph) -> UniversalIR {
    // Build node ID to index mapping
    let id_to_index: HashMap<String, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| (node.id.clone(), idx))
        .collect();

    // First pass: collect all edges for cycle detection
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for edge in &graph.edges {
        if let (Some(&from_idx), Some(&to_idx)) = (id_to_index.get(&edge.from), id_to_index.get(&edge.to)) {
            edges.push((from_idx, to_idx));
        }
    }

    // Auto-detect cycles BEFORE building successor/predecessor lists
    let (auto_loop_headers, back_edges) = detect_cycles(graph.nodes.len(), &edges);

    // First, determine which nodes are "pure backedge blocks"
    // A node is a pure backedge block if ALL its outgoing edges are back edges (no forward edges)
    let mut is_backedge_block = vec![false; graph.nodes.len()];
    for idx in 0..graph.nodes.len() {
        let outgoing_edges: Vec<_> = edges.iter().filter(|&&(from, _)| from == idx).collect();
        if outgoing_edges.is_empty() {
            continue;
        }
        // Check if ALL outgoing edges are back edges
        let all_are_back_edges = outgoing_edges.iter().all(|&&(from, to)| {
            from == to || back_edges.contains(&(from, to))
        });
        if all_are_back_edges {
            is_backedge_block[idx] = true;
        }
    }

    // Second pass: build successor/predecessor/back_edge lists
    // Rules:
    // - Self-loops: tracked for rendering but not added to successors/predecessors
    // - Explicit backedge blocks: Keep their back edges in successors (layout algorithm needs this)
    // - Other back edges: stored in back_edges_list (for rendering), not in successors
    // - Forward edges: add to both successors and predecessors
    let mut successors: Vec<Vec<String>> = vec![Vec::new(); graph.nodes.len()];
    let mut predecessors: Vec<Vec<String>> = vec![Vec::new(); graph.nodes.len()];
    let mut back_edges_list: Vec<Vec<String>> = vec![Vec::new(); graph.nodes.len()];
    let mut has_self_loop: Vec<bool> = vec![false; graph.nodes.len()];

    for edge in &graph.edges {
        if let (Some(&from_idx), Some(&to_idx)) = (id_to_index.get(&edge.from), id_to_index.get(&edge.to)) {
            let to_id = graph.nodes[to_idx].id.clone();
            let from_id = graph.nodes[from_idx].id.clone();

            // Track self-loops for rendering
            if from_idx == to_idx {
                has_self_loop[from_idx] = true;
                continue;
            }

            let is_back_edge = back_edges.contains(&(from_idx, to_idx));
            // Check if source is a backedge block (either explicit attribute or auto-detected)
            let source_is_backedge_block = graph.nodes[from_idx].is_backedge || is_backedge_block[from_idx];

            if is_back_edge {
                if source_is_backedge_block {
                    // Backedge block (explicit or auto-detected): Keep the edge in successors (layout needs it)
                    // This is a proper loop structure where the layout algorithm handles rendering
                    if !successors[from_idx].contains(&to_id) {
                        successors[from_idx].push(to_id);
                    }
                    // Note: We don't add to predecessors so the loop header can still be a root
                } else {
                    // Auto-detected back edge from a non-backedge block (e.g., bidirectional):
                    // add to back_edges_list for separate rendering
                    if !back_edges_list[from_idx].contains(&to_id) {
                        back_edges_list[from_idx].push(to_id);
                    }
                }
            } else {
                // Forward edge: add to successors and predecessors
                if !successors[from_idx].contains(&to_id) {
                    successors[from_idx].push(to_id);
                }
                if !predecessors[to_idx].contains(&from_id) {
                    predecessors[to_idx].push(from_id);
                }
            }
        }
    }

    // Only mark nodes as loopheaders if there's a corresponding backedge block
    // (i.e., a node whose ONLY purpose is to jump back to the loop header)
    // If there's no backedge block, we're just filtering cycles and don't need loop handling
    let mut should_be_loopheader = vec![false; graph.nodes.len()];
    for (idx, &is_header) in auto_loop_headers.iter().enumerate() {
        if is_header {
            // Check if there's actually a backedge block pointing to this header
            // A backedge block is a block that ONLY has back edges (no forward successors)
            let has_backedge_block = graph.nodes.iter().enumerate().any(|(other_idx, _)| {
                is_backedge_block[other_idx] && back_edges.contains(&(other_idx, idx))
            });
            should_be_loopheader[idx] = has_backedge_block;
        }
    }

    // Convert nodes to blocks
    let blocks: Vec<UniversalBlock> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            node_to_block(
                node,
                &successors[idx],
                &predecessors[idx],
                &back_edges_list[idx],
                has_self_loop[idx],
                should_be_loopheader[idx],
                is_backedge_block[idx],
            )
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert(
        "name".to_string(),
        crate::json_compat::Value::String(graph.name.clone()),
    );

    UniversalIR {
        format: UNIVERSAL_VERSION.to_string(),
        compiler: "dot".to_string(),
        metadata,
        blocks,
    }
}

fn node_to_block(
    node: &DotNode,
    successors: &[String],
    predecessors: &[String],
    back_edges: &[String],
    has_self_loop: bool,
    auto_loop_header: bool,
    auto_backedge: bool,
) -> UniversalBlock {
    // Build attributes list (manual attributes take precedence, auto-detected add if not present)
    let mut attributes = Vec::new();
    if node.is_loop_header || auto_loop_header {
        attributes.push("loopheader".to_string());
    }
    if node.is_backedge || auto_backedge {
        attributes.push("backedge".to_string());
    }

    // Create instructions from label lines (split by \n)
    let label = node.label.clone().unwrap_or_else(|| node.id.clone());
    let instructions: Vec<UniversalInstruction> = label
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(|line| UniversalInstruction {
            opcode: line.trim().to_string(),
            attributes: Vec::new(),
            type_: None,
            profiling: None,
            metadata: HashMap::new(),
        })
        .collect();

    // Ensure at least one instruction (use node ID if label is empty)
    let instructions = if instructions.is_empty() {
        vec![UniversalInstruction {
            opcode: node.id.clone(),
            attributes: Vec::new(),
            type_: None,
            profiling: None,
            metadata: HashMap::new(),
        }]
    } else {
        instructions
    };

    UniversalBlock {
        id: node.id.clone(),
        attributes,
        loop_depth: 0, // Will be computed by layout algorithm
        predecessors: predecessors.to_vec(),
        successors: successors.to_vec(),
        back_edges: back_edges.to_vec(),
        has_self_loop,
        instructions,
        metadata: HashMap::new(),
    }
}
