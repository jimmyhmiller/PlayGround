/**
 * BenchmarkChart
 *
 * Displays benchmark results as a bar chart comparing execution times
 * of different code variants against a baseline.
 */

import { memo, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import type { BenchmarkVariantResult } from './benchmarkTypes';

interface BenchmarkChartProps {
  results: BenchmarkVariantResult[];
  baselineId: string;
}

/**
 * Get bar color based on performance vs baseline
 */
function getBarColor(variantId: string, baselineId: string, baselineMean: number, mean: number): string {
  if (variantId === baselineId) {
    return '#6366f1'; // Primary color for baseline
  }

  const diff = ((mean - baselineMean) / baselineMean) * 100;

  if (diff < -10) {
    return '#22c55e'; // Green - significantly faster
  } else if (diff > 10) {
    return '#ef4444'; // Red - significantly slower
  } else {
    return '#f59e0b'; // Yellow - similar
  }
}

/**
 * BenchmarkChart Component
 */
const BenchmarkChart = memo(function BenchmarkChart({
  results,
  baselineId,
}: BenchmarkChartProps) {
  const chartData = useMemo(() => {
    return results.map((r) => ({
      name: r.name,
      mean: r.stats.mean,
      id: r.id,
    }));
  }, [results]);

  const baselineResult = results.find((r) => r.id === baselineId);
  const baselineMean = baselineResult?.stats.mean ?? 0;

  if (results.length === 0) {
    return (
      <div className="benchmark-chart" style={{ textAlign: 'center', padding: '40px', color: 'var(--theme-text-muted)' }}>
        No results yet. Run a benchmark to see the chart.
      </div>
    );
  }

  return (
    <div className="benchmark-chart">
      <div className="benchmark-chart-title">Execution Time (ms)</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
          <XAxis type="number" stroke="var(--theme-text-muted)" fontSize={12} />
          <YAxis
            type="category"
            dataKey="name"
            stroke="var(--theme-text-muted)"
            fontSize={12}
            width={70}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--theme-bg-elevated)',
              border: '1px solid var(--theme-border-primary)',
              borderRadius: '4px',
              color: 'var(--theme-text-primary)',
            }}
            formatter={(value: number) => [`${value.toFixed(3)} ms`, 'Mean']}
          />
          {baselineMean > 0 && (
            <ReferenceLine
              x={baselineMean}
              stroke="var(--theme-accent-primary)"
              strokeDasharray="3 3"
              label={{
                value: 'baseline',
                position: 'top',
                fill: 'var(--theme-text-muted)',
                fontSize: 10,
              }}
            />
          )}
          <Bar dataKey="mean" radius={[0, 4, 4, 0]}>
            {chartData.map((entry) => (
              <Cell
                key={entry.id}
                fill={getBarColor(entry.id, baselineId, baselineMean, entry.mean)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

export default BenchmarkChart;
