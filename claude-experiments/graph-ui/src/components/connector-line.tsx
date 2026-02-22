import type { ConnectorLineProps } from '../types'

export function ConnectorLine({
  direction,
  color,
  length = 16,
}: ConnectorLineProps) {
  if (direction === 'vertical') {
    return (
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <div
          style={{
            width: 1,
            height: length,
            background: `linear-gradient(to bottom, ${color}50, ${color}15)`,
          }}
        />
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <div
        style={{
          height: 1,
          width: length,
          background: `linear-gradient(to right, ${color}15, ${color}50)`,
        }}
      />
    </div>
  )
}
