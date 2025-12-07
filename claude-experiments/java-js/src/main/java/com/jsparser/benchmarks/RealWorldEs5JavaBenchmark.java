package com.jsparser.benchmarks;

import com.jsparser.Parser;
import com.jsparser.ast.Program;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * Real-World JavaScript Library Benchmarks - ES5 Java Parsers
 *
 * Benchmarks parsing performance on ES5-transpiled JavaScript libraries.
 * Includes Rhino benchmarks on smaller files that it can handle.
 *
 * Only includes libraries that Rhino can successfully parse without stack overflow:
 * - React (16.8 KB ES5)
 * - Lodash (469.5 KB ES5)
 * - React-DOM (220.2 KB ES5)
 *
 * Compares: Our Parser vs Nashorn vs GraalJS vs Rhino (all on ES5 code)
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(0) // Run in same JVM for faster iteration
public class RealWorldEs5JavaBenchmark {

    // ES5 transpiled code (for all parsers including Rhino)
    private static String LODASH_ES5;
    private static String REACT_DOM_ES5;
    private static String REACT_ES5;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        // Load ES5 versions
        LODASH_ES5 = Files.readString(Path.of("benchmarks/real-world-libs-es5/lodash.es5.js"));
        REACT_DOM_ES5 = Files.readString(Path.of("benchmarks/real-world-libs-es5/react-dom.production.min.es5.js"));
        REACT_ES5 = Files.readString(Path.of("benchmarks/real-world-libs-es5/react.production.min.es5.js"));

        System.out.println("Loaded ES5 benchmark files (Rhino-compatible only):");
        System.out.println("  Lodash:     " + formatSize(LODASH_ES5.length()));
        System.out.println("  React-DOM:  " + formatSize(REACT_DOM_ES5.length()));
        System.out.println("  React:      " + formatSize(REACT_ES5.length()));
    }

    private String formatSize(int bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
    }

    // ========================================================================
    // Our Parser Benchmarks (ES5)
    // ========================================================================

    @Benchmark
    public Program ourParser_lodash_es5() {
        Parser parser = new Parser(LODASH_ES5);
        return parser.parse();
    }

    @Benchmark
    public Program ourParser_reactDom_es5() {
        Parser parser = new Parser(REACT_DOM_ES5);
        return parser.parse();
    }

    @Benchmark
    public Program ourParser_react_es5() {
        Parser parser = new Parser(REACT_ES5);
        return parser.parse();
    }

    // ========================================================================
    // Nashorn Parser Benchmarks (ES5)
    // ========================================================================

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_lodash_es5() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("lodash.es5.js", LODASH_ES5, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_reactDom_es5() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("react-dom.es5.js", REACT_DOM_ES5, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_react_es5() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("react.es5.js", REACT_ES5, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ========================================================================
    // Rhino Parser Benchmarks (ES5)
    // ========================================================================

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_lodash_es5() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(LODASH_ES5, "lodash.es5.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_reactDom_es5() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(REACT_DOM_ES5, "react-dom.es5.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_react_es5() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(REACT_ES5, "react.es5.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ========================================================================
    // GraalJS Parser Benchmarks (ES5)
    // ========================================================================

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_lodash_es5() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", LODASH_ES5, "lodash.es5.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_reactDom_es5() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", REACT_DOM_ES5, "react-dom.es5.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_react_es5() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", REACT_ES5, "react.es5.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // Main method to run benchmarks
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
