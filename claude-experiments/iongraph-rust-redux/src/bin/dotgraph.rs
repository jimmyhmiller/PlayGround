// DOT graph visualizer using IonGraph layout algorithm
//
// Usage:
//   dotgraph <input.dot> [output.svg]
//   dotgraph <input.dot> --html [output.html]

use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::compilers::dot::{parse_dot, dot_to_universal};
use iongraph_rust_redux::layout_provider::LayoutProvider;
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use std::env;
use std::fs;
use std::process;

fn print_usage(program_name: &str) {
    eprintln!("Usage:");
    eprintln!("  {} <input.dot> [output.svg]", program_name);
    eprintln!("  {} <input.dot> --html [output.html]", program_name);
    eprintln!();
    eprintln!("DOT file format:");
    eprintln!("  digraph MyGraph {{");
    eprintln!("    A [label=\"Entry Block\"]");
    eprintln!("    B [label=\"Loop Header\" loopheader=\"true\"]");
    eprintln!("    C [label=\"Loop Body\" backedge=\"true\"]");
    eprintln!("    A -> B");
    eprintln!("    B -> C");
    eprintln!("    C -> B  // backedge");
    eprintln!("  }}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 2 {
        print_usage(program_name);
        process::exit(1);
    }

    let dot_path = &args[1];

    // Check for --html flag
    let (output_path, html_mode) = if args.len() > 2 && args[2] == "--html" {
        let path = args.get(3).map(|s| s.as_str()).unwrap_or("output.html");
        (path, true)
    } else {
        let path = args.get(2).map(|s| s.as_str()).unwrap_or("output.svg");
        (path, false)
    };

    // Read DOT file
    let dot_content = fs::read_to_string(dot_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", dot_path, err);
        process::exit(1);
    });

    // Parse DOT file
    let dot_graph = parse_dot(&dot_content).unwrap_or_else(|err| {
        eprintln!("Error parsing DOT file: {}", err);
        process::exit(1);
    });

    eprintln!("Rendering DOT graph \"{}\"", dot_graph.name);
    eprintln!("  Nodes: {}", dot_graph.nodes.len());
    eprintln!("  Edges: {}", dot_graph.edges.len());

    // Convert to UniversalIR
    let universal_ir = dot_to_universal(&dot_graph);

    if html_mode {
        render_to_html(universal_ir, output_path, &dot_graph.name);
    } else {
        render_to_svg(universal_ir, output_path);
    }
}

fn render_to_svg(universal_ir: iongraph_rust_redux::compilers::universal::UniversalIR, output_path: &str) {
    // Create layout provider and graph
    let mut layout_provider = PureSVGTextLayoutProvider::new();

    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    // Create the graph from Universal IR
    let mut graph = Graph::new(layout_provider, universal_ir, options);

    // Build graph layout
    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();

    // Render to SVG
    graph.render(nodes_by_layer, layer_heights, track_heights);

    // Create root SVG and append graph_container to it
    layout_provider = graph.layout_provider;
    let mut svg_root = layout_provider.create_svg_element("svg");
    layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");

    // Add 40 pixels to width and height
    let width = (graph.size.x + 40.0).ceil() as i32;
    let height = (graph.size.y + 40.0).ceil() as i32;

    layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
    layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
    layout_provider.set_attribute(
        &mut svg_root,
        "viewBox",
        &format!("0 0 {} {}", width, height),
    );
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

fn render_to_html(universal_ir: iongraph_rust_redux::compilers::universal::UniversalIR, output_path: &str, title: &str) {
    // Create layout provider and graph
    let mut layout_provider = PureSVGTextLayoutProvider::new();

    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    // Create the graph from Universal IR
    let mut graph = Graph::new(layout_provider, universal_ir, options);

    // Build graph layout
    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();

    // Render to SVG
    graph.render(nodes_by_layer, layer_heights, track_heights);

    // Create root SVG and append graph_container to it
    layout_provider = graph.layout_provider;
    let mut svg_root = layout_provider.create_svg_element("svg");
    layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");

    let width = (graph.size.x + 40.0).ceil() as i32;
    let height = (graph.size.y + 40.0).ceil() as i32;

    layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
    layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
    layout_provider.set_attribute(
        &mut svg_root,
        "viewBox",
        &format!("0 0 {} {}", width, height),
    );
    layout_provider.append_child(&mut svg_root, graph.graph_container);

    let svg_content = layout_provider.to_svg_string(&svg_root);

    // Create simple HTML wrapper
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DotGraph - {}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        h1 {{
            color: #eee;
            margin-bottom: 20px;
        }}
        .container {{
            background: #f5f5f5;
            border-radius: 8px;
            padding: 20px;
            overflow: auto;
        }}
        svg {{
            display: block;
        }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div class="container">
        {}
    </div>
</body>
</html>"#,
        title, title, svg_content
    );

    // Write to file
    fs::write(output_path, html).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_path, err);
        process::exit(1);
    });

    eprintln!("✓ HTML generated: {}", output_path);
    eprintln!("  Dimensions: {}x{}", width, height);
}
