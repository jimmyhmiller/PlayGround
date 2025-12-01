use iongraph_rust::*;
use std::fs;

fn main() {
    let json_str = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/test-loop-exit.json")
        .expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    
    let func = &ion_json.functions[0];
    let pass = func.passes[0].clone();
    
    let mut graph = Graph::new(Vec2::new(1000.0, 1000.0), pass);
    println!("Done");
}
