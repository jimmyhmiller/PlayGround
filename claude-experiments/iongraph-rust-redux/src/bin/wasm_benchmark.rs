// WASI benchmark binary - measures WASM performance outside the browser

use iongraph_rust_redux::compilers::ion::schema::IonJSON;
use iongraph_rust_redux::compilers::universal::pass_to_universal;
use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use iongraph_rust_redux::layout_provider::LayoutProvider;
use std::time::Instant;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <ion-json-file> <func-idx> <pass-idx>", args[0]);
        std::process::exit(1);
    }

    let json_path = &args[1];
    let func_idx: usize = args[2].parse().expect("Invalid function index");
    let pass_idx: usize = args[3].parse().expect("Invalid pass index");

    // Read file
    let json_str = fs::read_to_string(json_path).expect("Failed to read file");
    println!("File size: {} bytes ({:.2} MB)", json_str.len(), json_str.len() as f64 / 1_000_000.0);

    // Benchmark JSON parsing
    let start = Instant::now();
    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str).expect("Failed to parse JSON");
    let parse_time = start.elapsed();
    println!("JSON parsing: {:?}", parse_time);

    // Get function and pass
    let func = &data.functions[func_idx];
    let pass = &func.passes[pass_idx];

    // Benchmark IR conversion
    let start = Instant::now();
    let universal_ir = pass_to_universal(pass, &func.name);
    let convert_time = start.elapsed();
    println!("IR conversion: {:?}", convert_time);

    // Benchmark graph creation + layout
    let start = Instant::now();
    let mut layout_provider = PureSVGTextLayoutProvider::new();
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };
    let mut graph = Graph::new(layout_provider, universal_ir, options);
    let create_time = start.elapsed();
    println!("Graph creation: {:?}", create_time);

    let start = Instant::now();
    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
    let layout_time = start.elapsed();
    println!("Layout computation: {:?}", layout_time);

    // Benchmark rendering
    let start = Instant::now();
    graph.render(nodes_by_layer, layer_heights, track_heights);
    let render_time = start.elapsed();
    println!("Graph rendering: {:?}", render_time);

    // Benchmark SVG serialization
    let start = Instant::now();
    layout_provider = graph.layout_provider;
    let mut svg_root = layout_provider.create_svg_element("svg");
    layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");
    let width = (graph.size.x + 40.0).ceil() as i32;
    let height = (graph.size.y + 40.0).ceil() as i32;
    layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
    layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
    layout_provider.set_attribute(&mut svg_root, "viewBox", &format!("0 0 {} {}", width, height));
    layout_provider.append_child(&mut svg_root, graph.graph_container);
    let svg_string = layout_provider.to_svg_string(&svg_root);
    let serialize_time = start.elapsed();
    println!("SVG serialization: {:?}", serialize_time);
    println!("SVG size: {} bytes ({:.2} MB)", svg_string.len(), svg_string.len() as f64 / 1_000_000.0);

    // Total time
    let total = parse_time + convert_time + create_time + layout_time + render_time + serialize_time;
    println!("\nTotal: {:?}", total);

    // Breakdown percentages
    let total_ms = total.as_secs_f64() * 1000.0;
    println!("\nBreakdown:");
    println!("  JSON parsing:     {:.1}%", (parse_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
    println!("  IR conversion:    {:.1}%", (convert_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
    println!("  Graph creation:   {:.1}%", (create_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
    println!("  Layout:           {:.1}%", (layout_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
    println!("  Rendering:        {:.1}%", (render_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
    println!("  SVG serialization:{:.1}%", (serialize_time.as_secs_f64() * 1000.0 / total_ms) * 100.0);
}
