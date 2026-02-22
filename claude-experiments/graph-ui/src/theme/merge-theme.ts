import type { GraphCanvasTheme } from '../types'
import { defaultTheme } from './default-theme'

/**
 * Deep merge a partial theme with the default theme.
 */
export function mergeTheme(partial?: Partial<GraphCanvasTheme>): GraphCanvasTheme {
  if (!partial) return defaultTheme

  return {
    ...defaultTheme,
    ...partial,
    edgeColors: {
      ...defaultTheme.edgeColors,
      ...partial.edgeColors,
    },
    arrowSize: partial.arrowSize ?? defaultTheme.arrowSize,
  }
}
