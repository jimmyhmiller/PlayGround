// Semantic component - always renders the same HTML structure
// Styling comes from agent-generated CSS

export interface MetricDisplayProps {
  value: string;
  label: string;
  unit?: string;
}

export function MetricDisplay({ value, label, unit }: MetricDisplayProps) {
  return (
    <div className="metric-display" data-unit={unit}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {unit && <span className="metric-unit">{unit}</span>}
    </div>
  );
}
