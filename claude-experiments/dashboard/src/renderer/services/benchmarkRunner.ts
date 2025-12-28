/**
 * BenchmarkRunner
 *
 * Service for running benchmarks across multiple code variants.
 * Uses the eval API to execute code and measure timing.
 */

import type {
  BenchmarkVariant,
  BenchmarkVariantResult,
  BenchmarkConfig,
  BenchmarkStats,
} from '../components/benchmarkTypes';
import { hashValue } from '../components/benchmarkTypes';

/**
 * Progress callback for benchmark runs
 */
export type BenchmarkProgressCallback = (
  completedVariants: number,
  totalVariants: number,
  currentVariant: string
) => void;

/**
 * Calculate statistics from an array of execution times
 */
function calculateStats(times: number[]): BenchmarkStats {
  if (times.length === 0) {
    return { mean: 0, median: 0, stdDev: 0, min: 0, max: 0 };
  }

  const sorted = [...times].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const mean = sum / sorted.length;
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const min = sorted[0] ?? 0;
  const max = sorted[sorted.length - 1] ?? 0;

  const squaredDiffs = sorted.map((t) => Math.pow(t - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / sorted.length;
  const stdDev = Math.sqrt(variance);

  return { mean, median, stdDev, min, max };
}

/**
 * Run benchmark for a single variant
 */
async function runVariantBenchmark(
  variant: BenchmarkVariant,
  config: BenchmarkConfig
): Promise<BenchmarkVariantResult> {
  const { iterations, warmupIterations } = config;

  // Warmup runs
  for (let i = 0; i < warmupIterations; i++) {
    await window.evalAPI.execute(variant.code, 'javascript');
  }

  // Actual benchmark runs
  const times: number[] = [];
  let lastResult: unknown = undefined;
  let success = true;
  let error: string | undefined;

  for (let i = 0; i < iterations; i++) {
    const result = await window.evalAPI.execute(variant.code, 'javascript');

    if (result.success) {
      times.push(result.executionTimeMs);
      lastResult = result.value;
    } else {
      success = false;
      error = result.error;
      break;
    }
  }

  const stats = calculateStats(times);
  const resultHash = hashValue(lastResult);

  return {
    id: variant.id,
    name: variant.name,
    stats,
    lastResult,
    resultHash,
    success,
    error,
  };
}

/**
 * Run benchmarks for all variants
 */
export async function runBenchmarks(
  variants: BenchmarkVariant[],
  config: BenchmarkConfig,
  onProgress?: BenchmarkProgressCallback
): Promise<BenchmarkVariantResult[]> {
  const results: BenchmarkVariantResult[] = [];

  for (let i = 0; i < variants.length; i++) {
    const variant = variants[i]!;

    onProgress?.(i, variants.length, variant.name);

    const result = await runVariantBenchmark(variant, config);
    results.push(result);
  }

  onProgress?.(variants.length, variants.length, 'Done');

  return results;
}

/**
 * Create a new variant with a unique ID
 */
export function createVariant(name: string, code: string): BenchmarkVariant {
  return {
    id: `variant-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    name,
    code,
  };
}

/**
 * Default variants for demonstration
 */
export const DEFAULT_VARIANTS: BenchmarkVariant[] = [
  createVariant('forEach', `
const arr = Array.from({ length: 1000 }, (_, i) => i);
let sum = 0;
arr.forEach(x => { sum += x; });
sum
`),
  createVariant('for loop', `
const arr = Array.from({ length: 1000 }, (_, i) => i);
let sum = 0;
for (let i = 0; i < arr.length; i++) {
  sum += arr[i];
}
sum
`),
  createVariant('reduce', `
const arr = Array.from({ length: 1000 }, (_, i) => i);
arr.reduce((acc, x) => acc + x, 0)
`),
];
