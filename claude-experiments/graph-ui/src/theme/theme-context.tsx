import { createContext, useContext, type ReactNode } from 'react'
import type { GraphCanvasTheme } from '../types'
import { defaultTheme } from './default-theme'

const GraphThemeContext = createContext<GraphCanvasTheme>(defaultTheme)

export function GraphThemeProvider({
  theme,
  children,
}: {
  theme: GraphCanvasTheme
  children: ReactNode
}) {
  return (
    <GraphThemeContext.Provider value={theme}>
      {children}
    </GraphThemeContext.Provider>
  )
}

export function useGraphTheme(): GraphCanvasTheme {
  return useContext(GraphThemeContext)
}
