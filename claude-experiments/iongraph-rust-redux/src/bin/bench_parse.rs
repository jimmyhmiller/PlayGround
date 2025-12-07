use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_parse <json_file> [iterations]");
        std::process::exit(1);
    }
    
    let input = &args[1];
    let iterations: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    
    let json_str = fs::read_to_string(input).unwrap();
    println!("File size: {} bytes", json_str.len());
    
    // Warm up
    let _: iongraph_rust_redux::json_compat::Value = 
        iongraph_rust_redux::json_compat::from_str(&json_str).unwrap();
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _: iongraph_rust_redux::json_compat::Value = 
            iongraph_rust_redux::json_compat::from_str(&json_str).unwrap();
    }
    let elapsed = start.elapsed();
    
    println!("Parsed {} iterations in {:?}", iterations, elapsed);
    println!("Average: {:.3}ms per parse", elapsed.as_secs_f64() / iterations as f64 * 1000.0);
}
