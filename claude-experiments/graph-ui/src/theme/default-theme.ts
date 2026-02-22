import type { GraphCanvasTheme } from '../types'

export const defaultTheme: GraphCanvasTheme = {
  background: '#050510',
  backgroundStars: true,
  vignette: true,
  edgeColors: {},
  edgeWidth: 1.5,
  edgeGlowWidth: 4,
  edgeOpacity: 0.5,
  edgeGlowOpacity: 0.08,
  edgeDashArray: '8 4',
  arrowSize: { width: 8, height: 5 },
  arrowOpacity: 0.7,
  panTransitionMs: 400,
  panTransitionEasing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
  controlsBackground: 'rgba(24, 24, 27, 0.8)',
  controlsBorder: '#3f3f46',
  controlsText: '#a1a1aa',
  controlsHoverBackground: '#27272a',
  controlsHoverText: '#d4d4d8',
  cursorDefault: 'default',
  cursorDragging: 'grabbing',
}
