use iongraph_rust::*;
use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <path-to-json> <function-index> [pass-index|\"all\"] [output.svg]", args[0]);
        eprintln!("  function-index: specific function number");
        eprintln!("  pass-index: specific pass number (default: 0)");
        eprintln!("  \"all\": render all passes vertically");
        eprintln!("Example: {} examples/mega-complex.json 5 0 output.svg", args[0]);
        eprintln!("Example: {} examples/mega-complex.json 5 all output.svg", args[0]);
        std::process::exit(1);
    }

    let data_path = &args[1];
    let func_index: usize = args[2].parse().expect("Function index must be a number");
    let pass_arg = args.get(3).map(|s| s.as_str()).unwrap_or("0");
    let output_path = args.get(4).map(|s| s.as_str()).unwrap_or("output.svg");

    // Read and parse JSON
    println!("Reading {}...", data_path);
    let json_str = fs::read_to_string(data_path).expect("Failed to read JSON file");

    println!("Parsing JSON...");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    if ion_json.functions.is_empty() {
        eprintln!("Error: JSON must contain at least one function");
        std::process::exit(1);
    }

    if func_index >= ion_json.functions.len() {
        eprintln!("Error: Function index {} out of range (max: {})", func_index, ion_json.functions.len() - 1);
        std::process::exit(1);
    }

    let func = &ion_json.functions[func_index];
    if func.passes.is_empty() {
        eprintln!("Error: Function must contain at least one pass");
        std::process::exit(1);
    }

    if pass_arg == "all" {
        // Render all passes vertically
        render_all_passes(func, func_index, output_path);
    } else {
        // Render single pass
        let pass_index: usize = pass_arg.parse().expect("Pass index must be a number or \"all\"");

        if pass_index >= func.passes.len() {
            eprintln!("Error: Pass index {} out of range (max: {})", pass_index, func.passes.len() - 1);
            std::process::exit(1);
        }

        let pass = func.passes[pass_index].clone();

        println!("Rendering function {} '{}', pass {}: '{}'",
            func_index, func.name, pass_index, pass.name);
        println!("  MIR blocks: {}", pass.mir.blocks.len());

        // Create graph and render SVG with larger viewport for complex graphs
        let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
        let svg = graph.render_svg();

        // Write output
        fs::write(output_path, &svg).expect("Failed to write SVG file");

        println!("✓ SVG generated: {}", output_path);
        println!("  Size: {:.2} KB", svg.len() as f64 / 1024.0);
    }
}

fn render_all_passes(func: &Func, _func_index: usize, output_path: &str) {
    println!("Rendering all {} passes for function '{}'", func.passes.len(), func.name);

    let pass_gap = 50.0;
    let mut total_height = pass_gap;
    let mut max_width = 0.0_f64;
    let mut pass_svgs = Vec::new();

    // Generate each pass
    for (i, pass) in func.passes.iter().enumerate() {
        println!("  Rendering pass {}: '{}'", i, pass.name);

        let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass.clone());
        let svg_content = graph.render_svg();

        // Extract the graph size from the SVG
        let width = if let Some(w) = extract_svg_dimension(&svg_content, "width") {
            w
        } else {
            1000.0
        };
        let height = if let Some(h) = extract_svg_dimension(&svg_content, "height") {
            h
        } else {
            1000.0
        };

        // Extract just the inner content (without the outer <svg> tag)
        let inner_content = extract_svg_inner_content(&svg_content);

        pass_svgs.push((i, pass.name.clone(), inner_content, width, height));
        max_width = f64::max(max_width, width);
        total_height += height + pass_gap;
    }

    total_height += pass_gap; // Extra padding at bottom

    // Create combined SVG
    let mut combined = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        max_width as i32, total_height as i32, max_width as i32, total_height as i32
    );

    let mut current_y = pass_gap;
    for (pass_index, pass_name, svg_content, _width, height) in pass_svgs {
        // Add pass label
        combined.push_str(&format!(
            "  <text x=\"10\" y=\"{}\" font-family=\"monospace\" font-size=\"14\" font-weight=\"bold\" fill=\"#0c0c0d\">Pass {}: {}</text>\n",
            current_y - 10.0, pass_index, pass_name
        ));
        combined.push_str(&format!("  <g transform=\"translate(0, {})\">\n", current_y));
        combined.push_str(&svg_content);
        combined.push_str("  </g>\n");
        current_y += height + pass_gap;
    }

    combined.push_str("</svg>");

    // Write output
    fs::write(output_path, &combined).expect("Failed to write SVG file");

    println!("✓ Combined SVG generated: {}", output_path);
    println!("  Passes rendered: {}", func.passes.len());
    println!("  Dimensions: {}x{}", max_width as i32, total_height as i32);
    println!("  Size: {:.2} KB", combined.len() as f64 / 1024.0);
}

fn extract_svg_dimension(svg: &str, attr: &str) -> Option<f64> {
    let pattern = format!("{}=\"", attr);
    if let Some(start) = svg.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = svg[value_start..].find('"') {
            return svg[value_start..value_start + end].parse().ok();
        }
    }
    None
}

fn extract_svg_inner_content(svg: &str) -> String {
    // Extract content between <svg...> and </svg>
    if let Some(start) = svg.find('>') {
        if let Some(end) = svg.rfind("</svg>") {
            return svg[start + 1..end].trim().to_string();
        }
    }
    svg.to_string()
}
