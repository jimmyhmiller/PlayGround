use iongraph_rust_redux::config::{Theme, LayoutConfig};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <theme-file> [layout-file]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} config/ion-default.toml", args[0]);
        eprintln!("  {} config/dark.toml config/layout-compact.toml", args[0]);
        process::exit(1);
    }

    let theme_file = &args[1];

    println!("Loading theme from: {}", theme_file);
    let theme = Theme::load(theme_file).unwrap_or_else(|err| {
        eprintln!("Error loading theme: {}", err);
        process::exit(1);
    });

    println!("\n=== Theme Configuration ===");
    println!("Name: {}", theme.config().metadata.name);
    if !theme.config().metadata.description.is_empty() {
        println!("Description: {}", theme.config().metadata.description);
    }
    if let Some(ref compiler) = theme.config().metadata.compiler {
        println!("Compiler: {}", compiler);
    }

    println!("\n=== Block Colors ===");
    println!("Header: {}", theme.block_header_color());
    println!("Loop Header: {}", theme.loop_header_color());
    println!("Backedge: {}", theme.backedge_color());

    println!("\n=== Instruction Attributes ===");
    if theme.config().instruction_attributes.is_empty() {
        println!("(none defined)");
    } else {
        for (attr, color) in &theme.config().instruction_attributes {
            println!("  {}: {}", attr, color);
        }
    }

    println!("\n=== Heatmap ===");
    println!("Enabled: {}", theme.heatmap().enabled);
    println!("Hot: {}", theme.heatmap().hot);
    println!("Cool: {}", theme.heatmap().cool);
    println!("Threshold: {}", theme.heatmap().threshold);

    println!("\n=== Arrows ===");
    println!("Normal: {}", theme.config().arrows.normal);
    println!("Backedge: {}", theme.config().arrows.backedge);
    println!("Loop Header: {}", theme.config().arrows.loop_header);

    // Load layout if provided
    if args.len() > 2 {
        let layout_file = &args[2];
        println!("\n\nLoading layout from: {}", layout_file);
        let layout = LayoutConfig::load(layout_file).unwrap_or_else(|err| {
            eprintln!("Error loading layout: {}", err);
            process::exit(1);
        });

        println!("\n=== Layout Configuration ===");
        println!("Block Margin X: {}", layout.blocks.margin_x);
        println!("Block Margin Y: {}", layout.blocks.margin_y);
        println!("Block Gap: {}", layout.blocks.gap);
        println!("Arrow Radius: {}", layout.arrows.radius);
        println!("Font: {} {}px", layout.text.font_family, layout.text.font_size);
    }

    println!("\n✓ Theme and layout loaded successfully!");
}
