package com.jsparser.benchmarks;

import com.jsparser.Parser;
import com.jsparser.ast.Program;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Comparative benchmarks between different JVM JavaScript parsers:
 * - Our custom parser (java-js-parser)
 * - Nashorn
 * - GraalVM JavaScript (GraalJS)
 *
 * Focuses on modern JavaScript features (ES6+)
 *
 * Quick run (recommended):
 *   ./run-benchmarks.sh ComparativeParserBenchmark
 *
 * Full benchmark run:
 *   ./run-benchmarks.sh ComparativeParserBenchmark -f 1 -wi 3 -i 5
 *
 * Manual run:
 *   mvn clean package -DskipTests
 *   java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class ComparativeParserBenchmark {

    // Test code samples (same as ParserBenchmark for fair comparison)
    private static final String SMALL_FUNCTION = """
        function add(a, b) {
            return a + b;
        }
        """;

    private static final String SMALL_CLASS = """
        function Calculator() {
            this.result = 0;
        }

        Calculator.prototype.add = function(a, b) {
            return a + b;
        };

        Calculator.prototype.subtract = function(a, b) {
            return a - b;
        };
        """;

    private static final String MEDIUM_ASYNC_MODULE = """
        function UserDataFetcher() {
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
        }
        """;

    private static final String LARGE_MODULE = """
        function DataProcessor(options) {
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
        };
        """;

    // === Our Parser Benchmarks ===

    @Benchmark
    public Program ourParser_SmallFunction() {
        Parser parser = new Parser(SMALL_FUNCTION, false);
        return parser.parse();
    }

    @Benchmark
    public Program ourParser_SmallClass() {
        Parser parser = new Parser(SMALL_CLASS, false);
        return parser.parse();
    }

    @Benchmark
    public Program ourParser_MediumAsyncModule() {
        Parser parser = new Parser(MEDIUM_ASYNC_MODULE, false);
        return parser.parse();
    }

    @Benchmark
    public Program ourParser_LargeModule() {
        Parser parser = new Parser(LARGE_MODULE, false);
        return parser.parse();
    }

    // === Nashorn Parser Benchmarks ===

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_SmallFunction() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("test.js", SMALL_FUNCTION, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_SmallClass() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("test.js", SMALL_CLASS, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_MediumAsyncModule() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("test.js", MEDIUM_ASYNC_MODULE, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.openjdk.nashorn.api.tree.CompilationUnitTree nashornParser_LargeModule() {
        try {
            org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
            return parser.parse("test.js", LARGE_MODULE, null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // === Rhino Parser Benchmarks ===
    // Mozilla Rhino is a mature JavaScript parser written in Java
    // Used for comparative benchmarking

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_SmallFunction() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(SMALL_FUNCTION, "test.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_SmallClass() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(SMALL_CLASS, "test.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_MediumAsyncModule() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(MEDIUM_ASYNC_MODULE, "test.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.mozilla.javascript.ast.AstRoot rhinoParser_LargeModule() {
        try {
            org.mozilla.javascript.CompilerEnvirons compilerEnv = new org.mozilla.javascript.CompilerEnvirons();
            compilerEnv.setRecordingComments(false);
            compilerEnv.setRecordingLocalJsDocComments(false);
            compilerEnv.setLanguageVersion(org.mozilla.javascript.Context.VERSION_ES6);

            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(compilerEnv);
            return parser.parse(LARGE_MODULE, "test.js", 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // === GraalJS Parser Benchmarks ===
    // Note: GraalJS parsing happens implicitly during Context.eval()
    // We measure the parsing overhead by evaluating but not executing

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_SmallFunction() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", SMALL_FUNCTION, "test.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_SmallClass() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", SMALL_CLASS, "test.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_MediumAsyncModule() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", MEDIUM_ASYNC_MODULE, "test.js").build();
            return context.parse(source);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public org.graalvm.polyglot.Value graalJS_LargeModule() {
        try (Context context = Context.newBuilder("js")
                .allowAllAccess(false)
                .option("engine.WarnInterpreterOnly", "false")
                .build()) {
            Source source = Source.newBuilder("js", LARGE_MODULE, "test.js").build();
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
