package com.jsparser.benchmarks;

import com.jsparser.Parser;
import com.jsparser.ast.Program;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

/**
 * Real-world JavaScript library benchmarks
 * Tests parser performance on actual production code:
 * - TypeScript compiler (8.6 MB)
 * - Three.js (1.3 MB)
 * - Lodash (531 KB)
 * - Vue 3 (130 KB)
 * - React DOM (129 KB)
 *
 * Run with: ./run-benchmarks.sh RealWorldBenchmark
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(1)
public class RealWorldBenchmark {

    private static final String LIBS_DIR = "benchmarks/real-world-libs/";

    private String reactCode;
    private String vueCode;
    private String reactDomCode;
    private String lodashCode;
    private String threeCode;
    private String typescriptCode;

    @Setup
    public void setup() throws IOException {
        reactCode = Files.readString(Paths.get(LIBS_DIR + "react.production.min.js"));
        vueCode = Files.readString(Paths.get(LIBS_DIR + "vue.global.prod.js"));
        reactDomCode = Files.readString(Paths.get(LIBS_DIR + "react-dom.production.min.js"));
        lodashCode = Files.readString(Paths.get(LIBS_DIR + "lodash.js"));
        threeCode = Files.readString(Paths.get(LIBS_DIR + "three.js"));
        typescriptCode = Files.readString(Paths.get(LIBS_DIR + "typescript.js"));
    }

    @Benchmark
    public Program parseReact() {
        Parser parser = new Parser(reactCode, false);
        return parser.parse();
    }

    @Benchmark
    public Program parseVue() {
        Parser parser = new Parser(vueCode, false);
        return parser.parse();
    }

    @Benchmark
    public Program parseReactDom() {
        Parser parser = new Parser(reactDomCode, false);
        return parser.parse();
    }

    @Benchmark
    public Program parseLodash() {
        Parser parser = new Parser(lodashCode, false);
        return parser.parse();
    }

    @Benchmark
    public Program parseThreeJs() {
        Parser parser = new Parser(threeCode, false);
        return parser.parse();
    }

    @Benchmark
    public Program parseTypeScript() {
        Parser parser = new Parser(typescriptCode, false);
        return parser.parse();
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
