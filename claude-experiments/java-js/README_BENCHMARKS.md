# JavaScript Parser Benchmarks

This project includes comprehensive benchmarking infrastructure for comparing JavaScript parsers across multiple languages.

## Quick Start

```bash
# Quick cross-language benchmark (fastest)
./quick-cross-lang-bench.sh

# Java/JVM parsers only
./run-benchmarks.sh ComparativeParserBenchmark

# Full rigorous cross-language benchmark
./run-all-benchmarks.sh
```

## What's Included

### 🔵 Java/JVM Parsers
- **Our Parser** - Hand-written Java parser
- **Rhino** - Mozilla JavaScript (mature, fast)
- **Nashorn** - Oracle JavaScript
- **GraalJS** - GraalVM polyglot

### 🟡 JavaScript Parsers
- **Meriyah** - Fastest JS parser
- **Esprima** - Reference implementation
- **Acorn** - Popular, widely used
- **@babel/parser** - Babel ecosystem

### 🟢 Rust Parsers
- **OXC** - The Oxidation Compiler (fastest overall)

## Latest Results Summary

**Performance Rankings (Average Rank across all tests):**

1. 🥇 **OXC (Rust)** - 0.4-12 µs ⚡ *Fastest*
2. 🥈 **Rhino (Java)** - 0.8-22 µs (1.8-2.2x slower than OXC)
3. 🥉 **Our Parser (Java)** - 0.9-40 µs (2-3x slower than OXC)
4. **Meriyah (JS)** - 1.5-42 µs
5. **Esprima (JS)** - 2.7-56 µs
6. **Nashorn (Java)** - 8-49 µs
7. **Acorn (JS)** - 4-64 µs
8. **@babel/parser (JS)** - 4-57 µs
9. **GraalJS (Java)** - 246-686 µs ⚠️ *Slowest*

### Key Insights

✅ **Our Parser Performance:**
- **Solid 3rd place** overall across all languages
- **Beats all JavaScript parsers** (except sometimes Meriyah)
- **Only 2-3x slower than the fastest parser (OXC Rust)**
- **Competitive with Rhino**, the industry-standard JVM parser

✅ **What This Means:**
- Our hand-written Java parser is **production-ready** for performance
- Performance is **excellent** considering it's a simple implementation
- Still **plenty of room for optimization** (profiling, JIT-friendly patterns)
- Proves Java can compete with Rust for parsing (with the right techniques)

## Detailed Documentation

- **[CROSS_LANG_BENCHMARKS.md](CROSS_LANG_BENCHMARKS.md)** - Full cross-language results
- **[BENCHMARKS.md](BENCHMARKS.md)** - Java/JVM-only benchmarks

## Benchmark Infrastructure

### Files
- `run-benchmarks.sh` - Java-only benchmarks
- `quick-cross-lang-bench.sh` - Fast cross-language benchmark
- `run-all-benchmarks.sh` - Full cross-language benchmark
- `analyze-benchmarks.py` - Result analysis and comparison
- `benchmarks/javascript/` - Node.js benchmark harness
- `benchmarks/rust/` - Rust benchmark harness

### Running Individual Benchmarks

```bash
# Java benchmarks
mvn clean package -DskipTests
java --enable-preview -jar target/benchmarks.jar

# JavaScript benchmarks
cd benchmarks/javascript && npm install && node benchmark.js

# Rust benchmarks
cd benchmarks/rust && cargo run --release
```

## JMH Options

```bash
# Quick test
./run-benchmarks.sh ParserBenchmark -f 0 -wi 1 -i 1

# Production benchmark
./run-benchmarks.sh ParserBenchmark -f 1 -wi 5 -i 10

# With profiling
./run-benchmarks.sh ParserBenchmark -f 1 -wi 3 -i 5 -prof gc
```

## Test Cases

All parsers tested on identical JavaScript code:

1. **Small Function** (40 chars) - Simple function declaration
2. **Small Class** (183 chars) - Prototype-based class
3. **Medium Async Module** (1507 chars) - Async/await patterns
4. **Large Module** (2673 chars) - Complex event-driven module

## Why These Results Matter

### For Production Use
- **Our parser is fast enough** for most use cases
- **Predictable performance** (no GraalJS-like initialization delays)
- **JVM benefits** (mature GC, excellent profiling tools)

### For Future Optimization
- **Clear baseline** established
- **Multiple implementation strategies** to study (Rust's OXC, Rhino's approach)
- **Benchmark-driven development** enabled

### For Language Comparison
- **Rust is fastest** (OXC) but our Java is competitive
- **JavaScript parsers in JavaScript** are surprisingly fast (Meriyah)
- **JVM startup overhead** matters less than algorithm quality

## Next Steps

1. ✅ Cross-language benchmarks - **COMPLETE**
2. Profile our Java parser to find optimization opportunities
3. Implement low-hanging fruit optimizations
4. Benchmark against real-world code (Test262, npm packages)
5. Add to CI/CD for regression testing
6. Consider GraalVM native-image for startup improvements

---

**Note:** All benchmarks run on Apple M1. Times will vary by hardware. Use the
relative comparisons (vs Fastest column) for meaningful analysis.
