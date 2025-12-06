package com.jsparser;

import com.jsparser.ast.Program;

public class ProfilingBenchmark {
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

    public static void main(String[] args) {
        System.out.println("=== Profiling Benchmark (30 seconds) ===");
        System.out.println("Run with: -agentpath:/path/to/libasyncProfiler.so=start,event=cpu,file=flamegraph.html");
        System.out.println();

        // Warmup
        System.out.println("Warming up...");
        for (int i = 0; i < 5000; i++) {
            Parser parser = new Parser(MEDIUM_MODULE, false);
            parser.parse();
        }
        System.out.println("Warmup complete. Starting profiling run...");
        System.out.println();

        // Long run for profiling
        long start = System.nanoTime();
        long count = 0;
        long durationNanos = 30_000_000_000L; // 30 seconds

        while (System.nanoTime() - start < durationNanos) {
            for (int i = 0; i < 1000; i++) {
                Parser parser = new Parser(MEDIUM_MODULE, false);
                Program program = parser.parse();
                count++;
            }

            // Print progress every 5 seconds
            long elapsed = (System.nanoTime() - start) / 1_000_000_000L;
            if (elapsed % 5 == 0 && elapsed > 0) {
                System.out.printf("Progress: %d seconds, %d iterations\n", elapsed, count);
            }
        }

        long end = System.nanoTime();
        double totalSeconds = (end - start) / 1_000_000_000.0;
        double avgMicros = (end - start) / (double) count / 1000.0;

        System.out.println();
        System.out.printf("Complete: %d iterations in %.2f seconds\n", count, totalSeconds);
        System.out.printf("Average: %.2f μs per parse\n", avgMicros);
        System.out.printf("Throughput: %.0f parses/second\n", count / totalSeconds);
    }
}
