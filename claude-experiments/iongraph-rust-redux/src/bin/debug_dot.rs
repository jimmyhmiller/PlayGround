// Debug tool to see what DOT converts to
use iongraph_rust_redux::compilers::dot::{parse_dot, dot_to_universal};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let dot_path = &args[1];

    let dot_content = fs::read_to_string(dot_path).unwrap();
    let dot_graph = parse_dot(&dot_content).unwrap();
    let universal_ir = dot_to_universal(&dot_graph);

    println!("Graph: {}", dot_graph.name);
    println!("Blocks: {}", universal_ir.blocks.len());
    println!();

    for block in &universal_ir.blocks {
        println!("Block {}:", block.id);
        println!("  attributes: {:?}", block.attributes);
        println!("  successors: {:?}", block.successors);
        println!("  predecessors: {:?}", block.predecessors);
        println!();
    }
}
