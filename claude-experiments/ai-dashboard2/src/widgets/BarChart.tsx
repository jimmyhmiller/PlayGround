import { FC, useMemo } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import { useWidgetData } from '../hooks/useWidgetData';

interface BarChartConfig {
  id: string;
  type: 'barChart' | 'bar-chart';
  label: string | string[];
  data?: number[] | { value: number }[] | { values: number[] };
  dataSource?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const BarChart: FC<BaseWidgetComponentProps> = ({ theme, config, reloadTrigger }) => {
  const barChartConfig = config as BarChartConfig;
  const { data, loading, error } = useWidgetData(barChartConfig, reloadTrigger);

  // Generate random data as fallback
  const randomBars = useMemo(() => Array.from({ length: 30 }, () => Math.floor(Math.random() * 80 + 20)), []);

  // Use loaded data if available, otherwise fall back to random data
  const bars = useMemo(() => {
    if (data) {
      // Support different data formats:
      // 1. Array of numbers: [20, 45, 60, ...]
      // 2. Array of objects: [{value: 20}, {value: 45}, ...]
      // 3. Object with values array: {values: [20, 45, 60, ...]}
      if (Array.isArray(data)) {
        return data.map((item: any) => typeof item === 'number' ? item : item.value);
      } else if ((data as any).values && Array.isArray((data as any).values)) {
        return (data as any).values;
      }
    }
    return randomBars;
  }, [data, randomBars]);

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        <span>{Array.isArray(barChartConfig.label) ? barChartConfig.label[0] : barChartConfig.label}</span>
        {Array.isArray(barChartConfig.label) && barChartConfig.label[1] && <span>{barChartConfig.label[1]}</span>}
      </div>
      {loading && (
        <div style={{ fontFamily: theme.textBody, color: theme.textColor, padding: '20px' }}>
          Loading data...
        </div>
      )}
      {error && (
        <div style={{ fontFamily: theme.textBody, color: theme.negative, padding: '20px' }}>
          Error: {error}
        </div>
      )}
      {!loading && !error && (
        <div className="chart-container">
          {bars.map((h, i) => (
            <div key={i} className="bar" style={{ height: `${h}%`, backgroundColor: theme.accent, borderRadius: theme.chartRadius }} />
          ))}
        </div>
      )}
    </>
  );
};
