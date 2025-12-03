use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::iongraph::IonJSON;
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use std::fs;

fn load_test_data(path: &str) -> IonJSON {
    let json_str = fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path));
    serde_json::from_str(&json_str).unwrap_or_else(|_| panic!("Failed to parse {}", path))
}

fn bench_complete_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_layout");
    group.sample_size(20);

    let ion_json = load_test_data("ion-examples/mega-complex.json");
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    for (func_idx, func) in ion_json.functions.iter().enumerate().take(10) {
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

fn bench_svg_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("svg_rendering");
    group.sample_size(20);

    let ion_json = load_test_data("ion-examples/mega-complex.json");
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    for (func_idx, func) in ion_json.functions.iter().enumerate().take(10) {
        if let Some(pass) = func.passes.first() {
            // Pre-compute layout
            let layout_provider = PureSVGTextLayoutProvider::new();
            let mut graph = Graph::new(layout_provider, pass.clone(), options.clone());
            let (nodes_by_layer, layer_heights, track_heights) = graph.layout();

            group.bench_with_input(
                BenchmarkId::new("function", func_idx),
                &(nodes_by_layer, layer_heights, track_heights),
                |b, (_nodes, _heights, _tracks)| {
                    b.iter(|| {
                        // Re-create graph to benchmark rendering only
                        let layout_provider = PureSVGTextLayoutProvider::new();
                        let mut graph = Graph::new(layout_provider, pass.clone(), options.clone());
                        let (n, h, t) = graph.layout();
                        graph.render(n, h, t);
                        black_box(());
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(20);

    let ion_json = load_test_data("ion-examples/mega-complex.json");
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    for (func_idx, func) in ion_json.functions.iter().enumerate().take(10) {
        if let Some(pass) = func.passes.first() {
            group.bench_with_input(BenchmarkId::new("function", func_idx), pass, |b, pass| {
                b.iter(|| {
                    let layout_provider = PureSVGTextLayoutProvider::new();
                    let mut graph =
                        Graph::new(layout_provider, black_box(pass.clone()), options.clone());
                    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
                    graph.render(nodes_by_layer, layer_heights, track_heights);
                    black_box(());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_complete_layout,
    bench_svg_rendering,
    bench_end_to_end
);
criterion_main!(benches);
