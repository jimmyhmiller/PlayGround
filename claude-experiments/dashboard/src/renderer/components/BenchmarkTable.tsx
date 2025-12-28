/**
 * BenchmarkTable
 *
 * Displays benchmark results in a tabular format with
 * statistics and comparison against baseline.
 */

import { memo, useMemo } from 'react';
import type { BenchmarkVariantResult } from './benchmarkTypes';

interface BenchmarkTableProps {
  results: BenchmarkVariantResult[];
  baselineId: string;
}

/**
 * Format a number with appropriate precision
 */
function formatMs(ms: number): string {
  if (ms < 0.001) {
    return '<0.001 ms';
  }
  if (ms < 1) {
    return `${ms.toFixed(3)} ms`;
  }
  if (ms < 10) {
    return `${ms.toFixed(2)} ms`;
  }
  return `${ms.toFixed(1)} ms`;
}

/**
 * Calculate percentage difference from baseline
 */
function calcDiff(value: number, baseline: number): { percent: number; label: string; className: string } {
  if (baseline === 0) {
    return { percent: 0, label: '-', className: '' };
  }

  const percent = ((value - baseline) / baseline) * 100;
  const sign = percent > 0 ? '+' : '';
  const label = `${sign}${percent.toFixed(1)}%`;

  let className = '';
  if (percent < -5) {
    className = 'faster';
  } else if (percent > 5) {
    className = 'slower';
  }

  return { percent, label, className };
}

/**
 * BenchmarkTable Component
 */
const BenchmarkTable = memo(function BenchmarkTable({
  results,
  baselineId,
}: BenchmarkTableProps) {
  const baselineResult = useMemo(
    () => results.find((r) => r.id === baselineId),
    [results, baselineId]
  );

  const baselineMean = baselineResult?.stats.mean ?? 0;
  const baselineHash = baselineResult?.resultHash ?? '';

  if (results.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '20px', color: 'var(--theme-text-muted)' }}>
        No results yet. Run a benchmark to see the table.
      </div>
    );
  }

  return (
    <table className="benchmark-table">
      <thead>
        <tr>
          <th>Variant</th>
          <th>Mean</th>
          <th>Median</th>
          <th>Std Dev</th>
          <th>Min</th>
          <th>Max</th>
          <th>vs Baseline</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody>
        {results.map((result) => {
          const isBaseline = result.id === baselineId;
          const diff = isBaseline
            ? { percent: 0, label: '-', className: '' }
            : calcDiff(result.stats.mean, baselineMean);

          const resultMatch =
            isBaseline || result.resultHash === baselineHash;

          return (
            <tr
              key={result.id}
              className={isBaseline ? 'baseline-row' : ''}
            >
              <td>
                {result.name}
                {isBaseline && (
                  <span
                    style={{
                      marginLeft: '8px',
                      fontSize: '0.75em',
                      background: 'var(--theme-accent-primary)',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '3px',
                    }}
                  >
                    baseline
                  </span>
                )}
              </td>
              <td>{formatMs(result.stats.mean)}</td>
              <td>{formatMs(result.stats.median)}</td>
              <td>{formatMs(result.stats.stdDev)}</td>
              <td>{formatMs(result.stats.min)}</td>
              <td>{formatMs(result.stats.max)}</td>
              <td className={diff.className}>{diff.label}</td>
              <td className={resultMatch ? 'match' : 'mismatch'}>
                {isBaseline ? '-' : resultMatch ? 'OK' : 'DIFFERS'}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
});

export default BenchmarkTable;
