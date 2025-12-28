/**
 * Benchmark Types
 *
 * Type definitions for benchmarking functionality.
 */

/**
 * A single benchmark variant (code version to compare)
 */
export interface BenchmarkVariant {
  id: string;
  name: string;
  code: string;
}

/**
 * Statistics for benchmark runs
 */
export interface BenchmarkStats {
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
}

/**
 * Result for a single variant
 */
export interface BenchmarkVariantResult {
  id: string;
  name: string;
  stats: BenchmarkStats;
  lastResult: unknown;
  resultHash: string;
  success: boolean;
  error?: string;
}

/**
 * Configuration for running benchmarks
 */
export interface BenchmarkConfig {
  iterations: number;
  warmupIterations: number;
  timeout: number;
}

/**
 * Complete benchmark state
 */
export interface BenchmarkState {
  variants: BenchmarkVariant[];
  results: BenchmarkVariantResult[];
  baselineId: string;
  activeVariantId: string;
  isRunning: boolean;
  config: BenchmarkConfig;
}

/**
 * Default benchmark configuration
 */
export const DEFAULT_BENCHMARK_CONFIG: BenchmarkConfig = {
  iterations: 100,
  warmupIterations: 10,
  timeout: 5000,
};

/**
 * Hash a value for comparison
 */
export function hashValue(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
