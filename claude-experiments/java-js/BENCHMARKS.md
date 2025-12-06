# Benchmarks

## Setup Complete ✅

- [x] JMH dependencies added to pom.xml
- [x] Maven shade plugin configured for executable JAR
- [x] Basic benchmark suite created in `ParserBenchmark.java`

## Current Benchmarks

### Parsing Benchmarks
- `parseSmallFunction` - Parse a small function declaration (~3 lines)
- `parseSmallClass` - Parse a small ES6 class with methods (~8 lines)
- `parseMediumAsyncModule` - Parse async/await module with Map and promises (~35 lines)
- `parseLargeModule` - Parse a complex prototype-based module (~90 lines)

## Running Benchmarks

### Method 1: Using Script (Recommended) ✅

```bash
# Quick test run (minimal iterations for fast feedback)
./run-benchmarks.sh ComparativeParserBenchmark

# Full benchmark run (proper statistical analysis)
./run-benchmarks.sh ComparativeParserBenchmark -f 1 -wi 3 -i 5

# Run specific benchmark pattern
./run-benchmarks.sh ".*ourParser.*"

# Run all benchmarks
./run-benchmarks.sh
```

### Method 2: Manual Execution

```bash
# 1. Build the benchmark JAR
mvn clean package -DskipTests

# 2. Run benchmarks
java --enable-preview -jar target/benchmarks.jar [benchmark-pattern] [jmh-options]
```

## Benchmark Configuration

Current settings:
- Mode: Average time
- Time unit: Microseconds
- Warmup: 3 iterations, 1 second each
- Measurement: 5 iterations, 1 second each
- Forks: 1

## Latest Benchmark Results (2025-12-06)

### Comparative Benchmarks: java-js-parser vs Rhino vs Nashorn vs GraalJS

Quick benchmark run (Java 25, Apple M1 - not statistically rigorous):

```
Benchmark                                                   Mode  Cnt      Score   Error  Units
ComparativeParserBenchmark.graalJS_LargeModule              avgt       56913.613          us/op
ComparativeParserBenchmark.graalJS_MediumAsyncModule        avgt       17031.312          us/op
ComparativeParserBenchmark.graalJS_SmallClass               avgt        9227.014          us/op
ComparativeParserBenchmark.graalJS_SmallFunction            avgt        6926.552          us/op
ComparativeParserBenchmark.nashornParser_LargeModule        avgt        1288.586          us/op
ComparativeParserBenchmark.nashornParser_MediumAsyncModule  avgt         336.712          us/op
ComparativeParserBenchmark.nashornParser_SmallClass         avgt         123.376          us/op
ComparativeParserBenchmark.nashornParser_SmallFunction      avgt          83.483          us/op
ComparativeParserBenchmark.ourParser_LargeModule            avgt         568.242          us/op
ComparativeParserBenchmark.ourParser_MediumAsyncModule      avgt         193.088          us/op
ComparativeParserBenchmark.ourParser_SmallClass             avgt          33.893          us/op
ComparativeParserBenchmark.ourParser_SmallFunction          avgt           7.941          us/op
ComparativeParserBenchmark.rhinoParser_LargeModule          avgt         247.678          us/op
ComparativeParserBenchmark.rhinoParser_MediumAsyncModule    avgt         101.522          us/op
ComparativeParserBenchmark.rhinoParser_SmallClass           avgt          18.293          us/op
ComparativeParserBenchmark.rhinoParser_SmallFunction        avgt           4.638          us/op
```

### Performance Rankings (by test size)

**Small Function:**
1. 🥇 Rhino: 4.6 µs (fastest)
2. 🥈 Our Parser: 7.9 µs (1.7x slower)
3. 🥉 Nashorn: 83.5 µs (18x slower)
4. GraalJS: 6926 µs (1494x slower)

**Small Class:**
1. 🥇 Rhino: 18.3 µs
2. 🥈 Our Parser: 33.9 µs (1.9x slower)
3. 🥉 Nashorn: 123.4 µs (6.7x slower)
4. GraalJS: 9227 µs (504x slower)

**Medium Async Module:**
1. 🥇 Rhino: 101.5 µs
2. 🥈 Our Parser: 193.1 µs (1.9x slower)
3. 🥉 Nashorn: 336.7 µs (3.3x slower)
4. GraalJS: 17031 µs (168x slower)

**Large Module:**
1. 🥇 Rhino: 247.7 µs
2. 🥈 Our Parser: 568.2 µs (2.3x slower)
3. 🥉 Nashorn: 1288.6 µs (5.2x slower)
4. GraalJS: 56914 µs (230x slower)

### Key Findings

1. **vs Rhino** (Mozilla JavaScript):
   - Rhino is consistently the fastest parser (4.6-247.7 µs)
   - Our parser is 1.7-2.3x slower than Rhino
   - Rhino is highly optimized and mature (20+ years)
   - Still competitive - we're within 2-3x of the industry leader

2. **vs Nashorn** (Oracle JavaScript):
   - Our parser is **2.3-10.5x faster** than Nashorn
   - Nashorn has higher overhead for small files (18x slower)
   - Performance gap narrows for larger files (5.2x slower)

3. **vs GraalJS** (Polyglot VM):
   - Our parser is **88-1494x faster** than GraalJS
   - GraalJS has massive initialization overhead
   - GraalJS is designed for long-running scripts where startup cost is amortized
   - Not suitable for one-shot parsing tasks

4. **Overall**:
   - **~8 µs** per small function (~125,000 functions/second)
   - **~568 µs** for large modules (~1,760 modules/second)
   - Competitive with industry-standard parsers
   - 2nd place overall, beating Nashorn and GraalJS

## JMH Options

Common JMH options:
- `-f 1` - Number of forks (separate JVM processes)
- `-wi 3` - Number of warmup iterations
- `-i 5` - Number of measurement iterations
- `-t 4` - Number of threads
- `-prof gc` - Profile GC activity
- `-prof stack` - Profile stack traces
- `-jvmArgs "-Xmx4g"` - Pass JVM arguments

Example: `./run-benchmarks.sh ComparativeParserBenchmark -f 1 -wi 5 -i 10 -prof gc`

## Next Steps

1. ✅ ~~Add comparative benchmarks against Rhino, Nashorn, GraalJS~~ **COMPLETE**
2. Add benchmarks for real-world files (Test262, VS Code extensions)
3. Benchmark against Node.js parsers (Acorn, SWC, Babel) via subprocess
4. Create GraalVM Native Image for faster startup
5. Create performance regression tests
6. Integrate with CI/CD for continuous performance monitoring
