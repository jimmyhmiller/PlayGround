package com.jsparser.benchmarks;

import com.jsparser.Parser;
import com.jsparser.ast.Program;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple benchmark for cross-language comparison.
 *
 * Methodology (consistent with Rust/JS benchmarks):
 *   1. Load all test files into memory
 *   2. For each file: warmup iterations, then measurement iterations
 *   3. Report average parsing time (excludes startup/I/O)
 *
 * Usage:
 *   java SimpleBenchmark [warmup_iterations] [measurement_iterations]
 *   java SimpleBenchmark 5 10    # 5 warmup, 10 measurement (default)
 *   java SimpleBenchmark 10 50   # 10 warmup, 50 measurement
 */
public class SimpleBenchmark {

    private static final String LIBS_DIR = "benchmarks/real-world-libs/";
    private static final int DEFAULT_WARMUP_ITERATIONS = 5;
    private static final int DEFAULT_MEASUREMENT_ITERATIONS = 10;

    private static int warmupIterations = DEFAULT_WARMUP_ITERATIONS;
    private static int measurementIterations = DEFAULT_MEASUREMENT_ITERATIONS;

    record Library(String name, String filename, String code) {
        int sizeKB() {
            return code.length() / 1024;
        }
    }

    record BenchmarkResult(String parser, double avgMillis, double throughputKBPerMs) {}

    public static void main(String[] args) throws IOException {
        // Parse command line arguments
        if (args.length >= 1) {
            warmupIterations = Integer.parseInt(args[0]);
        }
        if (args.length >= 2) {
            measurementIterations = Integer.parseInt(args[1]);
        }
        System.out.println();
        System.out.println("═".repeat(80));
        System.out.println("Real-World JavaScript Library Benchmarks (Java)");
        System.out.println("═".repeat(80));
        System.out.println("Warmup iterations: " + warmupIterations);
        System.out.println("Measurement iterations: " + measurementIterations);
        System.out.println("═".repeat(80));

        List<Library> libraries = List.of(
            loadLibrary("React", "react.production.min.js"),
            loadLibrary("Vue 3", "vue.global.prod.js"),
            loadLibrary("React DOM", "react-dom.production.min.js"),
            loadLibrary("Lodash", "lodash.js"),
            loadLibrary("Three.js", "three.js"),
            loadLibrary("TypeScript Compiler", "typescript.js")
        );

        for (Library lib : libraries) {
            runBenchmark(lib);
        }

        System.out.println();
        System.out.println("═".repeat(80));
        System.out.println("Benchmark complete!");
        System.out.println("═".repeat(80));
    }

    private static Library loadLibrary(String name, String filename) throws IOException {
        String code = Files.readString(Paths.get(LIBS_DIR + filename));
        return new Library(name, filename, code);
    }

    private static void runBenchmark(Library lib) {
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("Library: " + lib.name);
        System.out.println("Size: " + lib.sizeKB() + " KB (" + lib.code.length() + " bytes)");
        System.out.println("=".repeat(80));
        System.out.println();

        // Warmup
        System.out.println("Warming up Our Java Parser...");
        for (int i = 0; i < warmupIterations; i++) {
            Parser parser = new Parser(lib.code, false);
            parser.parse();
        }

        // Measurement
        System.out.println("Benchmarking Our Java Parser...");
        long start = System.nanoTime();
        for (int i = 0; i < measurementIterations; i++) {
            Parser parser = new Parser(lib.code, false);
            parser.parse();
        }
        long elapsed = System.nanoTime() - start;

        double avgMillis = (elapsed / 1_000_000.0) / measurementIterations;
        double throughput = lib.sizeKB() / avgMillis;

        System.out.println();
        System.out.println("Results:");
        System.out.println("-".repeat(80));
        System.out.printf("%-20s | %15s | %20s%n", "Parser", "Avg Time (ms)", "Throughput (KB/ms)");
        System.out.println("-".repeat(80));
        System.out.printf("🥇 %-18s | %15.3f | %20.1f%n", "Our Java Parser", avgMillis, throughput);
    }
}
