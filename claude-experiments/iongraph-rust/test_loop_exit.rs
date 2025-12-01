use iongraph_rust::*;
use std::fs;

fn main() {
    let json_str = fs::read_to_string("../../../open-source/iongraph2/test-loop-exit.json")
        .expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    
    let func = &ion_json.functions[0];
    let pass = func.passes[0].clone();
    
    let mut graph = Graph::new(Vec2::new(1000.0, 1000.0), pass);
    
    println!("Block layer assignments:");
    for block in &graph.blocks {
        println!("Block {}: layer={}, loopDepth={}, loopHeight={}", 
            block.number.0, block.layer, block.loop_depth, block.loop_height);
    }
}
