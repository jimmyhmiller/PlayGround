//! Headless verification: feed one or more .jsonl files (or a directory) to the
//! OTel loader and print summary stats. Useful for sanity-checking the loader
//! without spinning up the GPU viewer.
//!
//! Usage: cargo run -p flame-format-otel --example smoke -- <path>...

use std::path::{Path, PathBuf};

use flame_core::{ProfileBuilder, TraceSource};
use flame_format_otel::OtelSource;

fn collect_files(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if p.is_dir() {
        for entry in std::fs::read_dir(p)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("jsonl"))
                    .unwrap_or(false)
            {
                out.push(path);
            }
        }
    } else if p.is_file() {
        out.push(p.to_path_buf());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: smoke <file-or-dir>...");
        std::process::exit(2);
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for a in &args {
        collect_files(Path::new(a), &mut files).expect("scan path");
    }
    files.sort();
    if files.is_empty() {
        eprintln!("no .jsonl files found");
        std::process::exit(1);
    }

    let mut buf = Vec::new();
    for f in &files {
        let bytes = std::fs::read(f).expect("read file");
        buf.extend_from_slice(&bytes);
        if !bytes.ends_with(b"\n") {
            buf.push(b'\n');
        }
        println!("  + {} ({} bytes)", f.display(), bytes.len());
    }

    let mut builder = ProfileBuilder::new();
    OtelSource.load(&buf, &mut builder).expect("load");
    let profile = builder.finish();

    println!("\nspans:        {}", profile.slices.len());
    println!("traces:       {}", profile.tracks.len());
    println!(
        "duration:     {:.3}s",
        profile.duration_ns() as f64 / 1e9
    );
    let max_depth = profile.slices.depth.iter().copied().max().unwrap_or(0);
    println!("max depth:    {}", max_depth);
    println!("services:");
    for c in &profile.categories {
        let name = profile.strings.get(c.name);
        if name == "default" {
            continue;
        }
        println!("  - {name}");
    }
    println!("\nfirst 10 traces by emit order:");
    for t in profile.tracks.iter().take(10) {
        println!(
            "  rows={} {}",
            t.row_count,
            profile.strings.get(t.name)
        );
    }
}
