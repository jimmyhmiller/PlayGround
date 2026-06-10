// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Dev tool: read a graph in a simple line format on stdin, run the layout
// core, dump cell geometry on stdout. Used by scripts/parity.ts to diff this
// port against the original essence.ts implementation.
//
// Input lines:
//   node <width> <height> <loop_depth> <loop_header:0|1> <backedge:0|1>
//   edge <tail_index> <head_index>
// Output lines (one per node, in input order):
//   cell <index> <left> <top> <width> <height> <layer>
// plus:
//   size <graph_width> <graph_height>

use std::io::Read;

use ion_layout::core::{layout, NodeSpec};

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");

    let mut nodes: Vec<NodeSpec> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (lineno, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "node" => {
                assert!(parts.len() == 6, "line {}: node needs 5 fields", lineno + 1);
                nodes.push(NodeSpec {
                    width: parts[1].parse().expect("width"),
                    height: parts[2].parse().expect("height"),
                    loop_depth: parts[3].parse().expect("loop_depth"),
                    loop_header: parts[4] == "1",
                    backedge: parts[5] == "1",
                });
            }
            "edge" => {
                assert!(parts.len() == 3, "line {}: edge needs 2 fields", lineno + 1);
                edges.push((parts[1].parse().expect("tail"), parts[2].parse().expect("head")));
            }
            other => panic!("line {}: unknown directive {other:?}", lineno + 1),
        }
    }

    let r = layout(&nodes, &edges);
    for i in 0..nodes.len() {
        let (w, h) = r.node_sizes[i];
        let p = r.positions[i];
        println!("cell {} {} {} {} {} {}", i, p.x - w / 2.0, p.y - h / 2.0, w, h, r.node_layers[i]);
    }
    println!("size {} {}", r.width, r.height);
}
