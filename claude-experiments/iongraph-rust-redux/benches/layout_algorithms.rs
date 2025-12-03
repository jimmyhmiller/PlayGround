use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::iongraph::IonJSON;
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use std::fs;

fn load_test_data(path: &str) -> IonJSON {
    let json_str = fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path));
    serde_json::from_str(&json_str).unwrap_or_else(|_| panic!("Failed to parse {}", path))
}

fn bench_layout_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_pipeline");

    let ion_json = load_test_data("ion-examples/mega-complex.json");
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    // Test first 5 functions, first pass of each
    for (func_idx, func) in ion_json.functions.iter().enumerate().take(5) {
        if let Some(pass) = func.passes.first() {
            group.bench_with_input(BenchmarkId::new("function", func_idx), pass, |b, pass| {
                b.iter(|| {
                    let layout_provider = PureSVGTextLayoutProvider::new();
                    let mut graph =
                        Graph::new(layout_provider, black_box(pass.clone()), options.clone());
                    black_box(graph.layout());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_layout_algorithm);
criterion_main!(benches);
