// ─── Types ───────────────────────────────────────────────
export type {
  NodePosition,
  NodeSize,
  CanvasEdge,
  EdgeSide,
  UseGraphCanvasOptions,
  GraphCanvasState,
  GraphCanvasActions,
  GraphCanvasHandlers,
  GraphCanvas,
  RenderNodeProps,
  GraphCanvasProps,
  GraphCanvasTheme,
  EdgeLayerProps,
  EdgePathProps,
  ConnectorLineProps,
  DefaultControlsProps,
} from './types'

// ─── Core Hook ───────────────────────────────────────────
export { useGraphCanvas } from './hooks/use-graph-canvas'

// ─── Sub-hooks ───────────────────────────────────────────
export { useResizeObserver } from './hooks/use-resize-observer'
export { useAnimatedPan } from './hooks/use-animated-pan'
export { useKeyboard } from './hooks/use-keyboard'

// ─── Components ──────────────────────────────────────────
export { GraphCanvas as GraphCanvasComponent } from './components/graph-canvas'
export { EdgeLayer } from './components/edge-layer'
export { EdgePath } from './components/edge-path'
export { DefaultControls } from './components/default-controls'
export { ConnectorLine } from './components/connector-line'

// ─── Layout Utilities ────────────────────────────────────
export { getEdgeSides } from './layout/edge-sides'
export { isOccupied } from './layout/collision'
export { computeExpandPosition, findFreePosition } from './layout/expand'

// ─── Theme ───────────────────────────────────────────────
export { defaultTheme } from './theme/default-theme'
export { GraphThemeProvider, useGraphTheme } from './theme/theme-context'
export { mergeTheme } from './theme/merge-theme'
