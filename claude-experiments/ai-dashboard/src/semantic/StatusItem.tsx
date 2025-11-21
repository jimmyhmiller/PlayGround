// Semantic status item

export interface StatusItemProps {
  label: string;
  value: string;
  state?: 'ok' | 'warn' | 'error';
}

export function StatusItem({ label, value, state = 'ok' }: StatusItemProps) {
  return (
    <div className="status-item" data-state={state}>
      <span className="status-label">{label}</span>
      <span className="status-value">{value}</span>
    </div>
  );
}
