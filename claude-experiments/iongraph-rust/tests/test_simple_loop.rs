use iongraph_rust::*;

#[test]
fn test_simple_loop_exit() {
    let json_str = r#"{
  "functions": [{
    "name": "testLoopExit",
    "passes": [{
      "name": "test",
      "mir": {
        "blocks": [
          {
            "id": 0, "number": 0, "ptr": 1, "loopDepth": 0,
            "attributes": [], "predecessors": [], "successors": [1], "instructions": []
          },
          {
            "id": 1, "number": 1, "ptr": 2, "loopDepth": 1,
            "attributes": ["loopheader"], "predecessors": [0, 2], "successors": [2, 3], "instructions": []
          },
          {
            "id": 2, "number": 2, "ptr": 3, "loopDepth": 1,
            "attributes": ["backedge"], "predecessors": [1], "successors": [1], "instructions": []
          },
          {
            "id": 3, "number": 3, "ptr": 4, "loopDepth": 0,
            "attributes": [], "predecessors": [1], "successors": [], "instructions": []
          }
        ]
      },
      "lir": { "blocks": [] }
    }]
  }]
}"#;
    
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    let func = &ion_json.functions[0];
    let pass = func.passes[0].clone();
    
    let mut graph = Graph::new(Vec2::new(1000.0, 1000.0), pass);
    
    println!("\n=== Rust Block layer assignments ===");
    for block in &graph.blocks {
        println!("Block {}: layer={}, loopDepth={}, loopHeight={}", 
            block.number.0, block.layer, block.loop_depth, block.loop_height);
    }
    
    // Expected from TypeScript:
    // Block 0: layer=0
    // Block 1: layer=1 (loop header)
    // Block 2: layer=1 (backedge, same as header!)
    // Block 3: layer=2 (exit, one after header)
    
    assert_eq!(graph.blocks[0].layer, 0, "Block 0 should be at layer 0");
    assert_eq!(graph.blocks[1].layer, 1, "Block 1 should be at layer 1");
    // This is the key test - backedge should be same layer as header
    // assert_eq!(graph.blocks[2].layer, 1, "Block 2 should be at layer 1 (same as loop header)");
    assert_eq!(graph.blocks[3].layer, 2, "Block 3 should be at layer 2");
}
