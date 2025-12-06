# Benchmarking TODO

## 1. JMH Benchmark Suite ✅ COMPLETED
- [x] Add JMH dependencies to pom.xml
- [x] Configure maven-compiler-plugin for annotation processing
- [x] Add exec-maven-plugin for easy benchmark execution
- [x] Create benchmark package structure (com.jsparser.benchmarks)
- [x] Implement basic parse benchmarks
  - [x] Small files (< 1KB) - parseSmallFunction, parseSmallClass
  - [x] Medium files (1-10KB) - parseMediumReactComponent
  - [x] Large files (> 10KB) - parseLargeModule
  - [ ] Test262 sample files (TODO)
- [x] Document how to run benchmarks (see BENCHMARKS.md)
- [x] Generate and verify results

**Results**: ~1 µs for small functions, ~38 µs for large modules (~25,000 functions/second)

## 2. Comparative Benchmarks - JVM Parsers ✅ COMPLETED
- [x] Set up Rhino parser comparison
  - [x] Add Rhino dependency
  - [x] Create Rhino benchmark harness
  - [x] Compare parse times
- [x] Set up Nashorn parser comparison (SKIPPED - removed in Java 15+)
- [x] Set up GraalJS parser comparison
  - [x] Add GraalJS dependency (org.graalvm.polyglot)
  - [x] Create GraalJS benchmark harness
  - [x] Compare parse times
- [x] Create comparison report/charts
- [x] Fix test cases to use standard JavaScript (no JSX)

**Results**: Our parser is **14-428x faster than GraalJS**! Competitive with Rhino on ES5, but supports modern ES6+ features (async/await, classes) that Rhino cannot parse.

| Parser | Small Function | Small Class | Medium Async | Large Module |
|--------|----------------|-------------|--------------|--------------|
| **java-js-parser** | 1.0 µs | 2.3 µs | 17.9 µs | 43.8 µs |
| **Rhino** | 0.5 µs | ❌ ES6 | ❌ ES6 | ❌ ES6 |
| **GraalJS** | 432 µs | 439 µs | 577 µs | 596 µs |

## 3. Comparative Benchmarks - Other Parsers
- [ ] Node.js/Acorn benchmarks
  - [ ] Create Node.js benchmark harness
  - [ ] Measure parse times
- [ ] Rust parsers (SWC, oxc, Biome)
  - [ ] Research how to benchmark Rust parsers
  - [ ] Set up cross-language benchmark harness
  - [ ] Measure parse times
- [ ] Babel parser benchmarks
- [ ] Create comprehensive comparison report

## 4. GraalVM Native Image
- [ ] Install GraalVM
- [ ] Add native-image-maven-plugin to pom.xml
- [ ] Configure reflection/resources for native-image
- [ ] Build native image
- [ ] Measure startup time improvements
- [ ] Compare native vs JVM performance
- [ ] Document build process

## 5. Performance Optimization
- [ ] Profile hotspots using JMH profilers
- [ ] Optimize based on benchmark results
- [ ] Re-benchmark after optimizations
- [ ] Document performance improvements

## Test Files for Benchmarking
- Small: Simple function declarations
- Medium: React component files
- Large: Bundled library files (lodash, moment, etc.)
- Real-world: VS Code extension files
- Test262: Representative sample from test suite

## Metrics to Track
- Parse time (total)
- Lexer time
- Parser time
- Memory allocation
- Throughput (files/sec)
- Startup time (for native image)
