// Semantic data series - renders as plain divs
// Agent CSS determines if it's bars, lines, circles, etc.

export interface DataSeriesProps {
  points: number[];
  labels?: string[];
  max?: number;
}

export function DataSeries({ points, labels, max }: DataSeriesProps) {
  const maxValue = max || Math.max(...points);

  return (
    <div className="data-series">
      {points.map((value, index) => (
        <div
          key={index}
          className="data-point"
          data-index={index}
          data-value={value}
          data-label={labels?.[index]}
          style={{
            // Only set CSS variable for value, styling agent decides how to use it
            ['--value' as any]: (value / maxValue) * 100,
          }}
        />
      ))}
    </div>
  );
}
