# Cross-Language JavaScript Parser Benchmarks

Comprehensive benchmarks comparing JavaScript parsers across **3 languages**: Java/JVM, JavaScript, and Rust.

## Quick Start

```bash
# Quick benchmark (fast, for testing)
./quick-cross-lang-bench.sh

# Full benchmark (rigorous, for production)
./run-all-benchmarks.sh
```

## Parsers Tested

### Java/JVM Parsers
- **Our Parser** - Hand-written recursive descent parser in Java
- **Rhino** - Mozilla's JavaScript engine (mature, 20+ years)
- **Nashorn** - Oracle's JavaScript engine (deprecated but still used)
- **GraalJS** - GraalVM JavaScript (polyglot VM)

### JavaScript Parsers
- **Meriyah** - Optimized for speed
- **Esprima** - Reference implementation
- **Acorn** - Widely used, popular
- **@babel/parser** - Most popular (Babel ecosystem)

### Rust Parsers
- **OXC** - The Oxidation Compiler (claims to be fastest)

## Latest Results (Quick Benchmark)

### Performance Rankings

**Overall Winner by Average Rank:**
1. 🥇 **OXC (Rust)** - 1.00 avg rank
2. 🥈 **Rhino (Java)** - 2.00 avg rank
3. 🥉 **Our Parser (Java)** - 3.00 avg rank
4. **Meriyah (JS)** - 4.00 avg rank
5. **Esprima (JS)** - 5.50 avg rank
6. **Nashorn (Java)** - 6.50 avg rank
7. **Acorn (JS)** - 6.75 avg rank
8. **@babel/parser (JS)** - 7.25 avg rank
9. **GraalJS (Java)** - 9.00 avg rank

### Small Function (40 chars)

| Parser | Time (µs) | vs Fastest |
|--------|-----------|------------|
| 🥇 OXC (Rust) | 0.418 | 1.00x |
| 🥈 Rhino (Java) | 0.781 | 1.87x |
| 🥉 Our Parser (Java) | 0.948 | 2.27x |
| Meriyah (JS) | 1.493 | 3.57x |
| Esprima (JS) | 2.695 | 6.45x |
| Acorn (JS) | 4.292 | 10.27x |
| @babel/parser (JS) | 4.397 | 10.52x |
| Nashorn (Java) | 8.283 | 19.82x |
| GraalJS (Java) | 245.734 | 587.88x |

### Small Class (183 chars)

| Parser | Time (µs) | vs Fastest |
|--------|-----------|------------|
| 🥇 OXC (Rust) | 1.170 | 1.00x |
| 🥈 Rhino (Java) | 2.616 | 2.24x |
| 🥉 Our Parser (Java) | 3.780 | 3.23x |
| Meriyah (JS) | 4.143 | 3.54x |
| Esprima (JS) | 5.917 | 5.06x |
| Acorn (JS) | 8.427 | 7.20x |
| @babel/parser (JS) | 9.794 | 8.37x |
| Nashorn (Java) | 10.988 | 9.39x |
| GraalJS (Java) | 263.633 | 225.33x |

### Medium Async Module (1507 chars)

| Parser | Time (µs) | vs Fastest |
|--------|-----------|------------|
| 🥇 OXC (Rust) | 7.252 | 1.00x |
| 🥈 Rhino (Java) | 12.354 | 1.70x |
| 🥉 Our Parser (Java) | 23.054 | 3.18x |
| Meriyah (JS) | 26.807 | 3.70x |
| Nashorn (Java) | 30.876 | 4.26x |
| Esprima (JS) | 32.952 | 4.54x |
| Acorn (JS) | 36.287 | 5.00x |
| @babel/parser (JS) | 36.465 | 5.03x |
| GraalJS (Java) | 328.456 | 45.29x |

### Large Module (2673 chars)

| Parser | Time (µs) | vs Fastest |
|--------|-----------|------------|
| 🥇 OXC (Rust) | 11.912 | 1.00x |
| 🥈 Rhino (Java) | 21.738 | 1.82x |
| 🥉 Our Parser (Java) | 39.683 | 3.33x |
| Meriyah (JS) | 42.255 | 3.55x |
| Nashorn (Java) | 49.241 | 4.13x |
| Esprima (JS) | 56.488 | 4.74x |
| @babel/parser (JS) | 57.001 | 4.79x |
| Acorn (JS) | 63.511 | 5.33x |
| GraalJS (Java) | 686.258 | 57.61x |

## Key Findings

### 🥇 OXC (Rust) - The Speed King
- **Fastest overall** across all test cases
- **0.4-12 µs** parse times
- 2-10x faster than our Java parser
- Pure Rust implementation with zero-copy parsing
- Production-ready and actively maintained

### 🥈 Rhino (Java) - The Veteran Champion
- **Consistently 2nd place** across all tests
- **0.8-22 µs** parse times
- Only **1.8-2.2x slower** than OXC
- 20+ years of optimization
- Mature and battle-tested

### 🥉 Our Parser (Java) - Strong Showing
- **Solid 3rd place** overall
- **0.9-40 µs** parse times
- **2-3x slower** than OXC, but very competitive
- **Beats all JavaScript parsers** except Meriyah
- Hand-written code without advanced optimizations
- **Plenty of room for optimization**

### Meriyah (JS) - Best JavaScript Parser
- **Best pure-JavaScript parser**
- **1.5-42 µs** parse times
- Specifically optimized for parsing performance
- Slightly slower than our Java parser

### GraalJS - The Outlier
- **Slowest by far** (245-686 µs)
- 45-588x slower than OXC
- Massive initialization overhead
- Not suitable for one-shot parsing
- Designed for long-running scripts

## Performance Insights

### Why is OXC so fast?

1. **Rust's zero-cost abstractions** - No runtime overhead
2. **Memory efficiency** - Arena allocators, zero-copy parsing
3. **SIMD optimizations** - Leverages CPU vector instructions
4. **Aggressive optimizations** - Designed for maximum performance

### Why is our Java parser competitive?

1. **Simple, focused implementation** - No unnecessary abstractions
2. **JIT compilation benefits** - HotSpot optimizes hot paths
3. **Efficient memory usage** - Minimal allocations
4. **Good algorithm choice** - Recursive descent is naturally fast

### Where can we improve?

1. **Lexer optimization** - Could use SIMD or table-driven approach
2. **Object pooling** - Reduce allocation overhead
3. **AST node efficiency** - More compact representations
4. **Benchmark-guided optimization** - Profile and optimize hot paths

## Benchmark Infrastructure

### Directory Structure

```
benchmarks/
├── javascript/          # Node.js benchmarks
│   ├── package.json
│   ├── benchmark.js
│   └── test-data.js
└── rust/               # Rust benchmarks
    ├── Cargo.toml
    └── src/main.rs

benchmark-results/       # Output directory
├── java-results.json
├── javascript-results.txt
└── rust-results.txt
```

### Scripts

- `quick-cross-lang-bench.sh` - Fast benchmark (1 iteration, for testing)
- `run-all-benchmarks.sh` - Full benchmark (5 iterations with warmup)
- `analyze-benchmarks.py` - Parse results and generate comparison tables
- `run-benchmarks.sh` - Java-only benchmarks

## Running Custom Benchmarks

### Java Only
```bash
./run-benchmarks.sh ComparativeParserBenchmark -f 1 -wi 3 -i 5
```

### JavaScript Only
```bash
cd benchmarks/javascript
npm install
node benchmark.js
```

### Rust Only
```bash
cd benchmarks/rust
cargo build --release
cargo run --release
```

### Custom Analysis
```bash
# After running benchmarks
python3 analyze-benchmarks.py
```

## Next Steps

1. ✅ Cross-language benchmarks - **COMPLETE**
2. Add more parsers (swc, esbuild via CLI)
3. Benchmark against real-world files (Test262, popular libraries)
4. Create performance regression tests
5. Implement optimizations based on findings
6. Continuous benchmarking in CI/CD

## Sources

Research on JavaScript parser performance:

- [GitHub - ecmascript-parser-benchmark](https://github.com/prantlf/ecmascript-parser-benchmark)
- [OXC Benchmarks](https://oxc.rs/docs/guide/benchmarks)
- [Benchmark TypeScript Parsers](https://medium.com/@hchan_nvim/benchmark-typescript-parsers-demystify-rust-tooling-performance-025ebfd391a3)
- [Esprima Speed Comparisons](https://esprima.org/test/compare.html)
