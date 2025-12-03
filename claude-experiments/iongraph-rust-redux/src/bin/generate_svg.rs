use std::env;
use std::fs;
use std::process;
use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use iongraph_rust_redux::layout_provider::LayoutProvider;
use iongraph_rust_redux::iongraph::IonJSON;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <path-to-json> <function-index> [pass-index] [output.svg]", args[0]);
        process::exit(1);
    }

    let json_path = &args[1];
    let func_idx: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Error: function-index must be a number");
        process::exit(1);
    });
    let pass_idx: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let output_path = args.get(4).map(|s| s.as_str()).unwrap_or("output.svg");

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let data: IonJSON = serde_json::from_str(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing JSON: {}", err);
        process::exit(1);
    });

    // Get the specified function and pass
    if func_idx >= data.functions.len() {
        eprintln!("Error: function index {} out of range (max: {})", func_idx, data.functions.len() - 1);
        process::exit(1);
    }

    let func = &data.functions[func_idx];

    if pass_idx >= func.passes.len() {
        eprintln!("Error: pass index {} out of range for function {} (max: {})", pass_idx, func_idx, func.passes.len() - 1);
        process::exit(1);
    }

    let pass = &func.passes[pass_idx];

    let func_name = &func.name;
    let pass_name = &pass.name;
    eprintln!("Rendering function {} \"{}\", pass {}: \"{}\"", func_idx, func_name, pass_idx, pass_name);

    if let Some(ref mir) = pass.mir {
        eprintln!("  MIR blocks: {}", mir.blocks.len());
    }

    // Create layout provider and graph
    let mut layout_provider = PureSVGTextLayoutProvider::new();

    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    // Create the graph (Graph::new calls build_blocks internally)
    let mut graph = Graph::new(layout_provider, pass.clone(), options);

    // Build graph layout
    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();

    // Render to SVG
    graph.render(nodes_by_layer, layer_heights, track_heights);

    // Create root SVG and append graph_container to it
    layout_provider = graph.layout_provider; // Take back the layout provider
    let mut svg_root = layout_provider.create_svg_element("svg");
    layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");

    // Add 40 pixels to width and height to match TypeScript (generate-svg-function.mjs:32-33)
    let width = (graph.size.x + 40.0).ceil() as i32;
    let height = (graph.size.y + 40.0).ceil() as i32;

    layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
    layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
    layout_provider.set_attribute(&mut svg_root, "viewBox", &format!("0 0 {} {}", width, height));
    layout_provider.append_child(&mut svg_root, graph.graph_container);

    // Get the SVG output
    let svg_output = layout_provider.to_svg_string(&svg_root);

    // Write to file
    fs::write(output_path, svg_output).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_path, err);
        process::exit(1);
    });

    eprintln!("✓ SVG generated: {}", output_path);
    eprintln!("  Dimensions: {}x{}", width, height);
}
