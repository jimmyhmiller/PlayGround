// Unified IonGraph binary - handles both Ion and Universal IR formats
//
// Usage:
//   iongraph <universal-json> [output.svg]                              # Universal IR (default)
//   iongraph --ion <ion-json> <function-index> [pass-index] [output.svg] # Ion format
//   iongraph --html <ion-json> [output.html]                            # HTML format (pre-rendered, ~18MB)
//   iongraph --wasm <ion-json> [output.html]                            # WASM format (client-side, ~1-2MB)
//   iongraph --viewer [output.html]                                     # WASM viewer (drag-and-drop, ~350KB)

use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::compilers::ion::schema::{IonJSON, Pass};
use iongraph_rust_redux::compilers::universal::{pass_to_universal, UniversalIR};
use iongraph_rust_redux::html_layout_provider::HTMLLayoutProvider;
use iongraph_rust_redux::html_templates::HTMLTemplate;
use iongraph_rust_redux::javascript_generator::JavaScriptGenerator;
use iongraph_rust_redux::layout_provider::LayoutProvider;
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use iongraph_rust_redux::wasm_html_generator::{generate_wasm_html, generate_wasm_viewer};
use std::env;
use std::fs;
use std::process;

fn print_usage(program_name: &str) {
    eprintln!("Usage:");
    eprintln!("  {} <universal-json> [output.svg]", program_name);
    eprintln!("  {} --ion <ion-json> <function-index> [pass-index] [output.svg]", program_name);
    eprintln!("  {} --html <ion-json> [output.html]        # Pre-rendered HTML (~18MB)", program_name);
    eprintln!("  {} --wasm <ion-json> [output.html]        # WASM-based HTML (~1-2MB)", program_name);
    eprintln!("  {} --viewer [output.html]                 # WASM viewer w/ drag-and-drop (~350KB)", program_name);
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} universal.json output.svg", program_name);
    eprintln!("  {} --ion mega-complex.json 0 0 output.svg", program_name);
    eprintln!("  {} --html mega-complex.json output.html    # Pre-rendered (large)", program_name);
    eprintln!("  {} --wasm mega-complex.json output.html    # Embedded JSON (medium)", program_name);
    eprintln!("  {} --viewer viewer.html                    # Drag-and-drop any JSON (tiny)", program_name);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 2 {
        print_usage(program_name);
        process::exit(1);
    }

    // Check for format flags
    match args[1].as_str() {
        "--ion" => handle_ion_format(&args),
        "--html" => handle_html_format(&args),
        "--wasm" => handle_wasm_format(&args),
        "--viewer" => handle_viewer_format(&args),
        _ => handle_universal_format(&args),
    }
}

fn handle_universal_format(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Error: Missing input file");
        print_usage(&args[0]);
        process::exit(1);
    }

    let json_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("output.svg");

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let universal_ir: UniversalIR = iongraph_rust_redux::json_compat::parse_as(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Universal IR JSON: {}", err);
        eprintln!("Hint: If this is an Ion format file, use the --ion flag");
        process::exit(1);
    });

    // Validate the universal format
    if let Err(e) = universal_ir.validate() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    let func_name = universal_ir.metadata.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    eprintln!("Rendering Universal IR function \"{}\"", func_name);
    eprintln!("  Compiler: {}", universal_ir.compiler);
    eprintln!("  Blocks: {}", universal_ir.blocks.len());

    render_to_svg(universal_ir, output_path);
}

fn handle_ion_format(args: &[String]) {
    // args[0] = program name
    // args[1] = --ion
    // args[2] = json path
    // args[3] = function index
    // args[4] = pass index (optional)
    // args[5] = output path (optional)

    if args.len() < 4 {
        eprintln!("Error: Ion format requires <ion-json> and <function-index>");
        print_usage(&args[0]);
        process::exit(1);
    }

    let json_path = &args[2];
    let func_idx: usize = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: function-index must be a number");
        process::exit(1);
    });
    let pass_idx: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let output_path = args.get(5).map(|s| s.as_str()).unwrap_or("output.svg");

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Ion JSON: {}", err);
        process::exit(1);
    });

    // Get the specified function and pass
    if func_idx >= data.functions.len() {
        eprintln!(
            "Error: function index {} out of range (max: {})",
            func_idx,
            data.functions.len() - 1
        );
        process::exit(1);
    }

    let func = &data.functions[func_idx];

    if pass_idx >= func.passes.len() {
        eprintln!(
            "Error: pass index {} out of range for function {} (max: {})",
            pass_idx,
            func_idx,
            func.passes.len() - 1
        );
        process::exit(1);
    }

    let pass = &func.passes[pass_idx];
    let func_name = &func.name;
    let pass_name = &pass.name;

    eprintln!(
        "Rendering Ion function {} \"{}\", pass {}: \"{}\"",
        func_idx, func_name, pass_idx, pass_name
    );

    if let Some(ref mir) = pass.mir {
        eprintln!("  MIR blocks: {}", mir.blocks.len());
    }

    // Convert Ion pass to Universal IR format
    let universal_ir = pass_to_universal(pass, func_name);
    eprintln!("  Converted to Universal IR: {} blocks", universal_ir.blocks.len());

    render_to_svg(universal_ir, output_path);
}

fn render_to_svg(universal_ir: UniversalIR, output_path: &str) {
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

fn handle_html_format(args: &[String]) {
    // args[0] = program name
    // args[1] = --html
    // args[2] = json path
    // args[3] = output path (optional)

    if args.len() < 3 {
        eprintln!("Error: HTML format requires <ion-json>");
        print_usage(&args[0]);
        process::exit(1);
    }

    let json_path = &args[2];
    let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("output.html");

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Ion JSON: {}", err);
        process::exit(1);
    });

    eprintln!(
        "Rendering HTML for {} functions",
        data.functions.len()
    );

    render_all_functions_to_html(&data, output_path);
}

fn render_all_passes_to_html(passes: &[Pass], func_name: &str, output_path: &str) {
    let mut graphs_html = String::new();
    let mut sidebar_html = String::new();

    // Find key passes (first MIR, last MIR, first LIR, last LIR)
    let mut key_passes: [Option<usize>; 4] = [None, None, None, None];
    let mut redundant_passes = Vec::new();
    let mut last_pass: Option<&Pass> = None;

    for (i, pass) in passes.iter().enumerate() {
        // Track key passes
        if pass.mir.is_some() {
            if key_passes[0].is_none() {
                key_passes[0] = Some(i); // First MIR
            }
            if pass.lir.is_none() {
                key_passes[1] = Some(i); // Last MIR (tentative)
            }
        }
        if pass.lir.is_some() {
            if last_pass.map(|p| p.lir.is_none()).unwrap_or(true) {
                key_passes[2] = Some(i); // First LIR
            }
            key_passes[3] = Some(i); // Last LIR (will be overwritten)
        }

        // Track redundant passes
        if let Some(prev) = last_pass {
            if passes_are_equal(prev, pass) {
                redundant_passes.push(i);
            }
        }
        last_pass = Some(pass);

        // Render this pass
        eprintln!("  Pass {}: {} ...", i, pass.name);

        let universal_ir = pass_to_universal(pass, func_name);
        let mut layout_provider = HTMLLayoutProvider::new();

        let options = GraphOptions {
            sample_counts: None,
            instruction_palette: None,
        };

        let mut graph = Graph::new(layout_provider, universal_ir, options);
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        graph.render(nodes_by_layer, layer_heights, track_heights);

        layout_provider = graph.layout_provider;
        let graph_html = graph.graph_container.to_html(0);

        // Wrap in graph container div
        let hidden_class = if i != 0 { " ig-hidden" } else { "" };
        graphs_html.push_str(&format!(
            r#"<div class="ig-graph{}" id="graph-{}" data-pass="{}">{}</div>"#,
            hidden_class, i, i, graph_html
        ));
        graphs_html.push('\n');

        // Generate sidebar entry
        let active_class = if i == 0 { " ig-active" } else { "" };
        let redundant_class = if redundant_passes.contains(&i) {
            " ig-redundant"
        } else {
            ""
        };
        sidebar_html.push_str(&format!(
            r#"<div class="ig-pass{}{}" data-pass="{}">{}: {}</div>"#,
            active_class, redundant_class, i, i, pass.name
        ));
        sidebar_html.push('\n');
    }

    // Generate JavaScript
    let js_generator = JavaScriptGenerator::new(passes.len())
        .with_key_passes(key_passes)
        .with_redundant_passes(redundant_passes);
    let javascript = js_generator.generate();

    // Create HTML template
    let template = HTMLTemplate::new(
        format!("IonGraph - {}", func_name),
        sidebar_html,
        graphs_html,
        javascript,
    );

    let html = template.render();

    // Write to file
    fs::write(output_path, html).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_path, err);
        process::exit(1);
    });

    eprintln!("✓ HTML generated: {}", output_path);
    eprintln!("  {} passes embedded", passes.len());
}

fn passes_are_equal(a: &Pass, b: &Pass) -> bool {
    // Simple equality check - could be more sophisticated
    a.name == b.name
        && a.mir.as_ref().map(|m| m.blocks.len()) == b.mir.as_ref().map(|m| m.blocks.len())
        && a.lir.as_ref().map(|l| l.blocks.len()) == b.lir.as_ref().map(|l| l.blocks.len())
}

fn render_all_functions_to_html(data: &IonJSON, output_path: &str) {
    let mut all_graphs_html = String::new();
    let mut all_sidebars_html = String::new();
    let mut function_selector_html = String::new();
    let mut total_passes = 0;

    // Build function selector dropdown
    function_selector_html.push_str(r#"<select id="function-selector" class="ig-function-selector">"#);
    function_selector_html.push('\n');

    for (func_idx, func) in data.functions.iter().enumerate() {
        let selected = if func_idx == 0 { " selected" } else { "" };
        function_selector_html.push_str(&format!(
            r#"  <option value="{}"{}>{}: {}</option>"#,
            func_idx, selected, func_idx, func.name
        ));
        function_selector_html.push('\n');

        eprintln!("  Function {}: \"{}\" ({} passes)", func_idx, func.name, func.passes.len());

        // Generate pass sidebar for this function
        let mut sidebar_html = String::new();
        let hidden_class = if func_idx != 0 { " ig-hidden" } else { "" };

        sidebar_html.push_str(&format!(
            r#"<div class="ig-pass-sidebar{}" data-function="{}">"#,
            hidden_class, func_idx
        ));
        sidebar_html.push('\n');

        // Find key passes and redundant passes for this function
        let mut key_passes: [Option<usize>; 4] = [None, None, None, None];
        let mut redundant_passes = Vec::new();
        let mut last_pass: Option<&Pass> = None;

        for (pass_idx, pass) in func.passes.iter().enumerate() {
            if pass.mir.is_some() {
                if key_passes[0].is_none() {
                    key_passes[0] = Some(pass_idx);
                }
                if pass.lir.is_none() {
                    key_passes[1] = Some(pass_idx);
                }
            }
            if pass.lir.is_some() {
                if last_pass.map(|p| p.lir.is_none()).unwrap_or(true) {
                    key_passes[2] = Some(pass_idx);
                }
                key_passes[3] = Some(pass_idx);
            }

            if let Some(prev) = last_pass {
                if passes_are_equal(prev, pass) {
                    redundant_passes.push(pass_idx);
                }
            }
            last_pass = Some(pass);

            let active_class = if pass_idx == 0 { " ig-active" } else { "" };
            let redundant_class = if redundant_passes.contains(&pass_idx) {
                " ig-redundant"
            } else {
                ""
            };
            sidebar_html.push_str(&format!(
                r#"  <div class="ig-pass{}{}" data-function="{}" data-pass="{}">{}: {}</div>"#,
                active_class, redundant_class, func_idx, pass_idx, pass_idx, pass.name
            ));
            sidebar_html.push('\n');
        }

        sidebar_html.push_str("</div>\n");
        all_sidebars_html.push_str(&sidebar_html);

        // Generate SVG graphs for all passes in this function
        for (pass_idx, pass) in func.passes.iter().enumerate() {
            let universal_ir = pass_to_universal(pass, &func.name);
            let mut layout_provider = PureSVGTextLayoutProvider::new();

            let options = GraphOptions {
                sample_counts: None,
                instruction_palette: None,
            };

            let mut graph = Graph::new(layout_provider, universal_ir, options);
            let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
            graph.render(nodes_by_layer, layer_heights, track_heights);

            // Get SVG output (reuse the validated SVG renderer!)
            layout_provider = graph.layout_provider;

            // Create root SVG
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

            // Get SVG string
            let svg_content = layout_provider.to_svg_string(&svg_root);

            // Wrap in div for HTML output
            let hidden_class = if func_idx != 0 || pass_idx != 0 { " ig-hidden" } else { "" };
            all_graphs_html.push_str(&format!(
                r#"<div class="ig-graph{}" id="graph-f{}-p{}" data-function="{}" data-pass="{}">{}</div>"#,
                hidden_class, func_idx, pass_idx, func_idx, pass_idx, svg_content
            ));
            all_graphs_html.push('\n');
            total_passes += 1;
        }
    }

    function_selector_html.push_str("</select>\n");

    // Generate JavaScript with multi-function support
    let js_generator = JavaScriptGenerator::new(total_passes)
        .with_multi_function_support(data.functions.len());
    let javascript = js_generator.generate();

    // Create HTML template
    let template = HTMLTemplate::new(
        "IonGraph Viewer".to_string(),
        format!("{}\n{}", function_selector_html, all_sidebars_html),
        all_graphs_html,
        javascript,
    );

    let html = template.render();

    fs::write(output_path, html).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_path, err);
        process::exit(1);
    });

    eprintln!("✓ HTML generated: {}", output_path);
    eprintln!("  {} functions, {} total passes", data.functions.len(), total_passes);
}

fn handle_wasm_format(args: &[String]) {
    // args[0] = program name
    // args[1] = --wasm
    // args[2] = json path
    // args[3] = output path (optional)

    if args.len() < 3 {
        eprintln!("Error: WASM format requires <ion-json>");
        print_usage(&args[0]);
        process::exit(1);
    }

    let json_path = &args[2];
    let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("output.html");

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Ion JSON: {}", err);
        process::exit(1);
    });

    eprintln!(
        "Generating WASM HTML for {} functions",
        data.functions.len()
    );

    // Generate WASM-based HTML
    generate_wasm_html(&data, output_path).unwrap_or_else(|err| {
        eprintln!("Error generating WASM HTML: {}", err);
        eprintln!();
        eprintln!("Make sure you have built the WASM binary first:");
        eprintln!("  wasm-pack build --target web --out-dir pkg");
        process::exit(1);
    });

    eprintln!("✓ WASM HTML generated: {}", output_path);
    eprintln!("  {} functions, client-side rendering enabled", data.functions.len());
    eprintln!();
    eprintln!("Note: This HTML file uses WASM for client-side rendering.");
    eprintln!("      Graphs will be rendered on-demand in the browser.");
}

fn handle_viewer_format(args: &[String]) {
    // args[0] = program name
    // args[1] = --viewer
    // args[2] = output path (optional)

    let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("iongraph-viewer.html");

    eprintln!("Generating standalone WASM viewer...");

    // Generate WASM viewer HTML
    generate_wasm_viewer(output_path).unwrap_or_else(|err| {
        eprintln!("Error generating WASM viewer: {}", err);
        eprintln!();
        eprintln!("Make sure you have built the WASM binary first:");
        eprintln!("  wasm-pack build --target web --out-dir pkg");
        process::exit(1);
    });

    eprintln!("✓ WASM viewer generated: {}", output_path);
    eprintln!();
    eprintln!("Features:");
    eprintln!("  • Drag-and-drop any Ion JSON file");
    eprintln!("  • Or click 'Choose File' to select a file");
    eprintln!("  • Client-side rendering with WASM");
    eprintln!("  • Works offline (WASM embedded)");
    eprintln!();
    eprintln!("Open in browser: open {}", output_path);
}
