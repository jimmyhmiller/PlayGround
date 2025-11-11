use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use pyret_attempt2::{Parser, Tokenizer, FileId};

/// Generate a large valid Pyret program with various constructs
fn generate_large_pyret_code(num_lines: usize) -> String {
    let mut code = String::new();

    // Add imports
    code.push_str("import lists as L\n");
    code.push_str("import equality as E\n");
    code.push_str("import sets as S\n\n");

    // Calculate how many constructs we need
    let lines_per_construct = 15; // Approximate lines per construct
    let num_constructs = num_lines / lines_per_construct;

    for i in 0..num_constructs {
        // Vary the constructs to make it realistic
        match i % 10 {
            0 => {
                // Simple function
                code.push_str(&format!("fun add{}(x, y):\n  x + y\nend\n\n", i));
            }
            1 => {
                // Function with where clause
                code.push_str(&format!(
                    "fun factorial{}(n):\n  if n == 0:\n    1\n  else:\n    n * factorial{}(n - 1)\n  end\nwhere:\n  factorial{}(5) is 120\nend\n\n",
                    i, i, i
                ));
            }
            2 => {
                // Data declaration
                code.push_str(&format!(
                    "data Tree{}:\n  | leaf(value)\n  | node(left :: Tree{}, right :: Tree{})\nend\n\n",
                    i, i, i
                ));
            }
            3 => {
                // For expression
                code.push_str(&format!(
                    "result{} = for map(x from [list: 1, 2, 3, 4, 5]):\n  x * 2\nend\n\n",
                    i
                ));
            }
            4 => {
                // Cases expression
                code.push_str(&format!(
                    "fun process{}(opt):\n  cases(Option) opt:\n    | some(v) => v * 2\n    | none => 0\n  end\nend\n\n",
                    i
                ));
            }
            5 => {
                // Lambda
                code.push_str(&format!(
                    "mapper{} = lam(x): x + {} end\n\n",
                    i, i
                ));
            }
            6 => {
                // Object literal
                code.push_str(&format!(
                    "point{} = {{\n  x: {},\n  y: {},\n  method distance(self, other):\n    dx = self.x - other.x\n    dy = self.y - other.y\n    num-sqrt((dx * dx) + (dy * dy))\n  end\n}}\n\n",
                    i, i, i
                ));
            }
            7 => {
                // Block with multiple statements
                code.push_str(&format!(
                    "value{} = block:\n  a = {}\n  b = a * 2\n  c = b + 3\n  c * 4\nend\n\n",
                    i, i
                ));
            }
            8 => {
                // Check block
                code.push_str(&format!(
                    "check \"test{}\":\n  {} + {} is {}\n  {} * {} is {}\nend\n\n",
                    i, i, 1, i + 1, i, 2, i * 2
                ));
            }
            _ => {
                // Complex nested expression
                code.push_str(&format!(
                    "complex{} = if {} > 0:\n  result = for fold(acc from 0, x from [list: 1, 2, 3]):\n    acc + x\n  end\n  result * {}\nelse:\n  0\nend\n\n",
                    i, i, i
                ));
            }
        }
    }

    code
}

/// Benchmark parsing different sized programs (string to AST)
fn bench_string_to_ast(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_to_ast");

    for size in [100, 1000, 10_000, 100_000].iter() {
        let code = generate_large_pyret_code(*size);
        let actual_lines = code.lines().count();

        group.throughput(Throughput::Elements(actual_lines as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_lines", actual_lines)),
            &code,
            |b, code| {
                b.iter(|| {
                    let file_id = FileId(0);
                    let mut tokenizer = Tokenizer::new(black_box(code), file_id);
                    let tokens = tokenizer.tokenize();
                    let mut parser = Parser::new(tokens, file_id);
                    black_box(parser.parse_program().unwrap())
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)  // Number of samples per benchmark
        .measurement_time(std::time::Duration::from_secs(10));  // Time per benchmark
    targets = bench_string_to_ast
}

criterion_main!(benches);
