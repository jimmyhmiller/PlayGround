/*!
 * Rust JavaScript Parser Benchmarks
 *
 * Compares performance of JavaScript parsers written in Rust:
 * - OXC (oxc_parser) - Claims to be the fastest JavaScript parser
 * - SWC (swc_ecma_parser) - Super fast TypeScript/JavaScript compiler
 */

use std::time::Instant;
use std::sync::Arc;

// OXC imports
use oxc_allocator::Allocator;
use oxc_parser::Parser as OxcParser;
use oxc_span::SourceType;

// SWC imports
use swc_common::SourceMap;
use swc_ecma_parser::{Parser as SwcParser, StringInput, Syntax, EsSyntax};

const SMALL_FUNCTION: &str = r#"function add(a, b) {
    return a + b;
}"#;

const SMALL_CLASS: &str = r#"function Calculator() {
    this.result = 0;
}

Calculator.prototype.add = function(a, b) {
    return a + b;
};

Calculator.prototype.subtract = function(a, b) {
    return a - b;
};"#;

const MEDIUM_ASYNC_MODULE: &str = r#"function UserDataFetcher() {
    this.cache = {};
}

UserDataFetcher.prototype.fetchUserData = function(userId, callback) {
    var self = this;

    if (this.cache[userId]) {
        callback(null, this.cache[userId]);
        return;
    }

    fetch('/api/users/' + userId).then(function(response) {
        if (!response.ok) {
            throw new Error('Failed to fetch user');
        }
        return response.json();
    }).then(function(data) {
        self.cache[userId] = data;
        callback(null, data);
    }).catch(function(error) {
        console.error('Error fetching user:', error);
        callback(error);
    });
};

UserDataFetcher.prototype.processUserBatch = function(userIds, callback) {
    var results = [];
    var completed = 0;
    var hasError = false;

    for (var i = 0; i < userIds.length; i++) {
        this.fetchUserData(userIds[i], function(err, data) {
            if (hasError) return;
            if (err) {
                hasError = true;
                callback(err);
                return;
            }
            results.push(data);
            completed++;
            if (completed === userIds.length) {
                callback(null, results);
            }
        });
    }
};

function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this;
        var args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}"#;

const LARGE_MODULE: &str = r#"function DataProcessor(options) {
    options = options || {};
    this.batchSize = options.batchSize || 100;
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 1000;
    this.queue = [];
    this.processing = false;
    this.listeners = {};
}

DataProcessor.prototype.on = function(event, handler) {
    if (!this.listeners[event]) {
        this.listeners[event] = [];
    }
    this.listeners[event].push(handler);
};

DataProcessor.prototype.emit = function(event, data) {
    var handlers = this.listeners[event] || [];
    for (var i = 0; i < handlers.length; i++) {
        handlers[i](data);
    }
};

DataProcessor.prototype.process = function(data, callback) {
    this.queue.push(data);
    this.emit('queued', { queueSize: this.queue.length });

    if (!this.processing) {
        this._processBatch(callback);
    }
};

DataProcessor.prototype._processBatch = function(callback) {
    var self = this;

    if (this.queue.length === 0) {
        this.processing = false;
        if (callback) callback();
        return;
    }

    this.processing = true;
    var batch = this.queue.splice(0, this.batchSize);
    var results = [];
    var completed = 0;

    for (var i = 0; i < batch.length; i++) {
        this._processItem(batch[i], function(err, result) {
            if (err) {
                self.emit('error', err);
                return;
            }
            results.push(result);
            completed++;
            if (completed === batch.length) {
                self.emit('batch-complete', { count: results.length });
                self._processBatch(callback);
            }
        });
    }
};

DataProcessor.prototype._processItem = function(item, callback) {
    var self = this;
    var attempt = 1;

    function tryProcess() {
        try {
            var hashValue = JSON.stringify(item);
            self._delay(10, function() {
                callback(null, {
                    id: item.id,
                    data: item.data,
                    hash: hashValue,
                    processed: true
                });
            });
        } catch (error) {
            if (attempt >= self.retryAttempts) {
                callback(error);
            } else {
                attempt++;
                self._delay(self.retryDelay * attempt, tryProcess);
            }
        }
    }

    tryProcess();
};

DataProcessor.prototype._delay = function(ms, callback) {
    setTimeout(callback, ms);
};

DataProcessor.prototype.getQueueSize = function() {
    return this.queue.length;
};

DataProcessor.prototype.isProcessing = function() {
    return this.processing;
};"#;

struct BenchmarkResult {
    name: String,
    avg_micros: f64,
}

/// Benchmark a parsing function
fn benchmark<F>(name: &str, mut parse_fn: F, code: &str, iterations: usize) -> BenchmarkResult
where
    F: FnMut(&str),
{
    // Warmup
    for _ in 0..1000 {
        parse_fn(code);
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        parse_fn(code);
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as f64 / iterations as f64;

    BenchmarkResult {
        name: name.to_string(),
        avg_micros,
    }
}

/// Parse with OXC
fn parse_with_oxc(code: &str) {
    let allocator = Allocator::default();
    let source_type = SourceType::default();
    let _ = OxcParser::new(&allocator, code, source_type).parse();
}

/// Parse with SWC
fn parse_with_swc(code: &str) {
    let cm = Arc::new(SourceMap::default());
    let fm = cm.new_source_file(swc_common::FileName::Anon.into(), code.to_string());
    let input = StringInput::from(&*fm);

    let syntax = Syntax::Es(EsSyntax {
        jsx: false,
        fn_bind: false,
        decorators: false,
        decorators_before_export: false,
        export_default_from: false,
        import_attributes: true,
        allow_super_outside_method: false,
        allow_return_outside_function: false,
        auto_accessors: false,
        explicit_resource_management: false,
    });

    let mut parser = SwcParser::new(syntax, input, None);
    let _ = parser.parse_module();
}

/// Run benchmark suite for a specific code sample
fn run_benchmark_suite(suite_name: &str, code: &str, iterations: usize) {
    println!("\n{}", "=".repeat(60));
    println!("Benchmark: {}", suite_name);
    println!("Code size: {} characters", code.len());
    println!("{}\n", "=".repeat(60));

    let mut results = vec![
        benchmark("OXC (Rust)", |c| parse_with_oxc(c), code, iterations),
        benchmark("SWC (Rust)", |c| parse_with_swc(c), code, iterations),
    ];

    // Sort by performance (fastest first)
    results.sort_by(|a, b| a.avg_micros.partial_cmp(&b.avg_micros).unwrap());

    // Print results
    println!("Results (sorted by performance):");
    println!("{}", "-".repeat(60));
    println!("{:<20} | {:>15} | {:>15}", "Parser", "Avg Time (µs)", "vs Fastest");
    println!("{}", "-".repeat(60));

    let fastest = results[0].avg_micros;

    for (i, result) in results.iter().enumerate() {
        let ratio = result.avg_micros / fastest;
        let indicator = match i {
            0 => "🥇",
            1 => "🥈",
            _ => "  ",
        };

        println!(
            "{} {:<18} | {:>15.3} | {:>15.2}x",
            indicator, result.name, result.avg_micros, ratio
        );
    }
}

fn main() {
    const ITERATIONS: usize = 10000;

    println!("\n{}", "═".repeat(60));
    println!("JavaScript Parser Benchmarks (Rust-based parsers)");
    println!("{}", "═".repeat(60));
    println!("Warmup iterations: 1000");
    println!("Measurement iterations: {}", ITERATIONS);
    println!("{}", "═".repeat(60));

    run_benchmark_suite("Small Function", SMALL_FUNCTION, ITERATIONS);
    run_benchmark_suite("Small Class", SMALL_CLASS, ITERATIONS);
    run_benchmark_suite("Medium Async Module", MEDIUM_ASYNC_MODULE, ITERATIONS);
    run_benchmark_suite("Large Module", LARGE_MODULE, ITERATIONS);

    println!("\n{}", "═".repeat(60));
    println!("Benchmark complete!");
    println!("{}", "═".repeat(60));
}
