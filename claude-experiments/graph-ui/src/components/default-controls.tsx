import type { DefaultControlsProps } from '../types'
import { mergeTheme } from '../theme/merge-theme'

const buttonStyle = (theme: ReturnType<typeof mergeTheme>): React.CSSProperties => ({
  borderRadius: 4,
  border: `1px solid ${theme.controlsBorder}`,
  background: theme.controlsBackground,
  padding: '4px 8px',
  fontSize: 10,
  color: theme.controlsText,
  cursor: 'pointer',
  lineHeight: 1,
})

export function DefaultControls({
  zoom,
  nodeCount,
  minZoom,
  maxZoom,
  onZoomIn,
  onZoomOut,
  onResetView,
  onClearGraph,
  theme: themeOverride,
}: DefaultControlsProps) {
  const theme = mergeTheme(themeOverride)

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 8,
        right: 12,
        zIndex: 20,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <span
        style={{
          fontFamily: 'monospace',
          fontSize: 9,
          color: theme.controlsText,
          opacity: 0.6,
        }}
      >
        {nodeCount} node{nodeCount !== 1 ? 's' : ''}
      </span>
      <button style={buttonStyle(theme)} onClick={onClearGraph}>
        Clear
      </button>
      <button style={buttonStyle(theme)} onClick={onResetView}>
        Reset view
      </button>
      <button style={buttonStyle(theme)} onClick={onZoomOut}>
        −
      </button>
      <span
        style={{
          fontFamily: 'monospace',
          fontSize: 9,
          color: theme.controlsText,
          opacity: 0.6,
        }}
      >
        {Math.round(zoom * 100)}%
      </span>
      <button style={buttonStyle(theme)} onClick={onZoomIn}>
        +
      </button>
    </div>
  )
}
