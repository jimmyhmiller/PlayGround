use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use paredit_like::{Parinfer, ClojureParser, Refactorer};
use std::fs;

fn generate_test_data() -> Vec<(String, String)> {
    vec![
        ("small".to_string(), "(defn foo [x] (+ x 1".to_string()),
        ("medium".to_string(), include_str!("../examples/unbalanced.clj").to_string()),
        ("large".to_string(), {
            let mut large = String::new();
            for i in 0..100 {
                large.push_str(&format!("(defn func{} [x{}] (+ x{} {}\n", i, i, i, i));
            }
            large
        }),
        ("deeply_nested".to_string(), {
            let mut nested = String::new();
            for _ in 0..50 {
                nested.push('(');
            }
            nested.push_str("inner");
            // Intentionally leave some unbalanced for parinfer to fix
            for _ in 0..40 {
                nested.push(')');
            }
            nested
        }),
        ("complex_let".to_string(), {
            let mut complex = String::new();
            for i in 0..20 {
                complex.push_str(&format!("(let [x{} {}] ", i, i));
            }
            complex.push_str("(+ ");
            for i in 0..20 {
                complex.push_str(&format!("x{} ", i));
            }
            complex.push(')');
            // Leave some unbalanced
            for _ in 0..15 {
                complex.push(')');
            }
            complex
        }),
    ]
}

fn bench_parinfer_balance(c: &mut Criterion) {
    let test_data = generate_test_data();
    
    let mut group = c.benchmark_group("parinfer_balance");
    for (name, input) in &test_data {
        group.bench_with_input(BenchmarkId::new("balance", name), input, |b, input| {
            b.iter(|| {
                let parinfer = Parinfer::new(black_box(input));
                black_box(parinfer.balance().unwrap())
            })
        });
    }
    group.finish();
}

fn bench_parser(c: &mut Criterion) {
    let test_data = generate_test_data();
    
    let mut group = c.benchmark_group("parser");
    for (name, input) in &test_data {
        group.bench_with_input(BenchmarkId::new("parse", name), input, |b, input| {
            b.iter(|| {
                let mut parser = ClojureParser::new().unwrap();
                black_box(parser.parse_to_sexpr(black_box(input)).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_refactoring(c: &mut Criterion) {
    let simple_cases = vec![
        ("slurp_simple", "(foo bar) baz"),
        ("barf_simple", "(foo bar baz)"),
        ("splice_simple", "(foo bar)"),
        ("wrap_simple", "foo bar"),
    ];
    
    let mut group = c.benchmark_group("refactoring");
    
    for (name, input) in &simple_cases {
        group.bench_with_input(BenchmarkId::new("slurp", name), input, |b, input| {
            b.iter(|| {
                let mut parser = ClojureParser::new().unwrap();
                let forms = parser.parse_to_sexpr(black_box(input)).unwrap();
                let mut refactorer = Refactorer::new(input.to_string());
                black_box(refactorer.slurp_forward(&forms, 1).unwrap_or_else(|_| input.to_string()))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("barf", name), input, |b, input| {
            b.iter(|| {
                let mut parser = ClojureParser::new().unwrap();
                let forms = parser.parse_to_sexpr(black_box(input)).unwrap();
                let mut refactorer = Refactorer::new(input.to_string());
                black_box(refactorer.barf_forward(&forms, 1).unwrap_or_else(|_| input.to_string()))
            })
        });
    }
    group.finish();
}

fn bench_complex_scenarios(c: &mut Criterion) {
    // Real-world like scenarios
    let scenarios = vec![
        ("namespace", r#"
(ns my.namespace
  (:require [clojure.string :as str]
            [clojure.set :as set]))

(defn process-data [data]
  (let [cleaned (remove nil? data)
        sorted (sort cleaned)
        unique (distinct sorted)]
    (vec unique
"#),
        ("nested_functions", r#"
(defn outer [x]
  (let [y (+ x 1)]
    (defn inner [z]
      (let [w (+ z y)]
        (fn [a] 
          (+ a w x
"#),
        ("map_operations", r#"
{:users [{:name "Alice" :age 30}
         {:name "Bob" :age 25}
         {:name "Charlie" :age 35}]
 :metadata {:created "2023-01-01"
           :version "1.0.0"
           :settings {:debug true
                     :log-level :info
"#),
    ];
    
    let mut group = c.benchmark_group("complex_scenarios");
    
    for (name, input) in &scenarios {
        group.bench_with_input(BenchmarkId::new("full_pipeline", name), input, |b, input| {
            b.iter(|| {
                // Full pipeline: parse -> balance -> parse again to verify
                let mut parser = ClojureParser::new().unwrap();
                let _forms = parser.parse_to_sexpr(black_box(input)).unwrap();
                
                let parinfer = Parinfer::new(black_box(input));
                let balanced = parinfer.balance().unwrap();
                
                let mut parser2 = ClojureParser::new().unwrap();
                black_box(parser2.parse_to_sexpr(&balanced).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_large_file(c: &mut Criterion) {
    // Generate a large file to test performance on realistic codebases
    let mut large_file = String::new();
    
    // Add namespace
    large_file.push_str("(ns large.file\n  (:require [clojure.string :as str]\n            [clojure.set :as set]))\n\n");
    
    // Add many function definitions with various nesting levels
    for i in 0..50 {
        large_file.push_str(&format!(r#"
(defn function-{} [arg1 arg2 arg3]
  (let [local1 (+ arg1 arg2)
        local2 (* arg2 arg3)
        local3 (- arg1 arg3)]
    (cond
      (> local1 100) {{:result :large
                      :value local1
                      :meta {{:processed true
                             :timestamp (System/currentTimeMillis)}}}}
      (< local2 10)  {{:result :small
                      :value local2
                      :meta {{:processed true
                             :notes ["small value detected"]}}}}
      :else          {{:result :medium
                      :value local3
                      :processing [{{:step 1 :action :validate}}
                                  {{:step 2 :action :transform}}
                                  {{:step 3 :action :output"#, i));
    }
    
    c.bench_function("large_file_balance", |b| {
        b.iter(|| {
            let parinfer = Parinfer::new(black_box(&large_file));
            black_box(parinfer.balance().unwrap())
        })
    });
    
    c.bench_function("large_file_parse", |b| {
        b.iter(|| {
            let mut parser = ClojureParser::new().unwrap();
            black_box(parser.parse_to_sexpr(black_box(&large_file)).unwrap())
        })
    });
}

fn bench_memory_intensive(c: &mut Criterion) {
    // Test with very deeply nested structures
    let mut deep_nesting = String::new();
    for _ in 0..200 {
        deep_nesting.push('(');
    }
    deep_nesting.push_str("innermost");
    for _ in 0..150 { // Leave some unbalanced
        deep_nesting.push(')');
    }
    
    c.bench_function("deep_nesting", |b| {
        b.iter(|| {
            let parinfer = Parinfer::new(black_box(&deep_nesting));
            black_box(parinfer.balance().unwrap())
        })
    });
    
    // Test with wide structures
    let mut wide_structure = String::new();
    wide_structure.push('(');
    for i in 0..1000 {
        wide_structure.push_str(&format!("item{} ", i));
    }
    // Leave unbalanced
    
    c.bench_function("wide_structure", |b| {
        b.iter(|| {
            let parinfer = Parinfer::new(black_box(&wide_structure));
            black_box(parinfer.balance().unwrap())
        })
    });
}

criterion_group!(
    benches,
    bench_parinfer_balance,
    bench_parser,
    bench_refactoring,
    bench_complex_scenarios,
    bench_large_file,
    bench_memory_intensive
);
criterion_main!(benches);