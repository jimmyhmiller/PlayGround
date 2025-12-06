package com.jsparser;

import com.jsparser.ast.Program;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.mozilla.javascript.CompilerEnvirons;

public class SimpleComparativeBenchmark {
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

    private static final String MEDIUM_MODULE = """
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

        DataProcessor.prototype.getQueueSize = function() {
            return this.queue.length;
        };
        """;

    public static void main(String[] args) {
        System.out.println("=== Comparative Parser Benchmark ===");
        System.out.println("Comparing: Our Parser vs Rhino vs Nashorn vs GraalJS");
        System.out.println();

        // Warmup all parsers
        System.out.println("Warming up...");
        warmup();
        System.out.println("Warmup complete.\n");

        // Run benchmarks
        System.out.println("Running benchmarks...\n");

        benchmarkTest("Small Function", SMALL_FUNCTION, 5000);
        System.out.println();

        benchmarkTest("Small Class", SMALL_CLASS, 3000);
        System.out.println();

        benchmarkTest("Medium Module", MEDIUM_MODULE, 2000);
        System.out.println();

        benchmarkTest("Large Module", LARGE_MODULE, 1000);
        System.out.println();

        System.out.println("=== Benchmark Complete ===");
    }

    private static void warmup() {
        for (int i = 0; i < 500; i++) {
            // Our parser
            new Parser(SMALL_FUNCTION, false).parse();

            // Rhino
            CompilerEnvirons env = new CompilerEnvirons();
            env.setRecoverFromErrors(true);
            new org.mozilla.javascript.Parser(env).parse(SMALL_FUNCTION, null, 1);

            // Nashorn
            try {
                org.openjdk.nashorn.api.tree.Parser.create().parse("test.js", SMALL_FUNCTION, null);
            } catch (Exception e) {}
        }

        // GraalJS warmup (smaller iterations - it's slow)
        for (int i = 0; i < 10; i++) {
            try (Context context = Context.newBuilder("js")
                    .allowAllAccess(false)
                    .option("engine.WarnInterpreterOnly", "false")
                    .build()) {
                Source source = Source.newBuilder("js", SMALL_FUNCTION, "test.js").build();
                context.parse(source);
            } catch (Exception e) {}
        }
    }

    private static void benchmarkTest(String name, String code, int iterations) {
        System.out.println("=== " + name + " ===");

        // Our Parser
        long ourStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            Parser parser = new Parser(code, false);
            Program program = parser.parse();
        }
        long ourEnd = System.nanoTime();
        double ourAvg = (ourEnd - ourStart) / (double) iterations / 1000.0;

        // Rhino
        long rhinoStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            CompilerEnvirons env = new CompilerEnvirons();
            env.setRecoverFromErrors(true);
            org.mozilla.javascript.Parser parser = new org.mozilla.javascript.Parser(env);
            parser.parse(code, null, 1);
        }
        long rhinoEnd = System.nanoTime();
        double rhinoAvg = (rhinoEnd - rhinoStart) / (double) iterations / 1000.0;

        // Nashorn
        long nashornStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            try {
                org.openjdk.nashorn.api.tree.Parser parser = org.openjdk.nashorn.api.tree.Parser.create();
                parser.parse("test.js", code, null);
            } catch (Exception e) {}
        }
        long nashornEnd = System.nanoTime();
        double nashornAvg = (nashornEnd - nashornStart) / (double) iterations / 1000.0;

        // GraalJS (fewer iterations - it's very slow)
        int graalIterations = Math.max(10, iterations / 10);
        long graalStart = System.nanoTime();
        for (int i = 0; i < graalIterations; i++) {
            try (Context context = Context.newBuilder("js")
                    .allowAllAccess(false)
                    .option("engine.WarnInterpreterOnly", "false")
                    .build()) {
                Source source = Source.newBuilder("js", code, "test.js").build();
                context.parse(source);
            } catch (Exception e) {}
        }
        long graalEnd = System.nanoTime();
        double graalAvg = (graalEnd - graalStart) / (double) graalIterations / 1000.0;

        // Print results
        System.out.printf("  Our Parser:  %7.2f μs (%d iterations)\n", ourAvg, iterations);
        System.out.printf("  Rhino:       %7.2f μs (%d iterations) - %.2fx %s\n",
            rhinoAvg, iterations, rhinoAvg / ourAvg, rhinoAvg < ourAvg ? "faster" : "slower");
        System.out.printf("  Nashorn:     %7.2f μs (%d iterations) - %.2fx %s\n",
            nashornAvg, iterations, nashornAvg / ourAvg, nashornAvg < ourAvg ? "faster" : "slower");
        System.out.printf("  GraalJS:     %7.2f μs (%d iterations) - %.2fx slower\n",
            graalAvg, graalIterations, graalAvg / ourAvg);
    }
}
