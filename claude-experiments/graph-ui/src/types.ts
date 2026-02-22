import { type ReactNode } from 'react'

// ─── Core Types ──────────────────────────────────────────

export interface NodePosition {
  x: number
  y: number
}

export interface NodeSize {
  w: number
  h: number
}

export interface CanvasEdge<Id = string> {
  fromId: Id
  toId: Id
  type: string
}

export type EdgeSide = 'top' | 'right' | 'bottom' | 'left'

// ─── Hook Options & State ────────────────────────────────

export interface UseGraphCanvasOptions {
  minZoom?: number
  maxZoom?: number
  spacing?: number
  collisionThreshold?: number
  edgeTypes?: Record<string, string>
  /** When true, Cmd/Meta must be held to pan or drag nodes. Default: true. */
  requireModifierKey?: boolean
}

export interface GraphCanvasState<Id = string> {
  nodes: Map<Id, NodePosition>
  edges: CanvasEdge<Id>[]
  selectedNodeId: Id | null
  graphRoot: Id | null
  pan: { x: number; y: number }
  zoom: number
  isDragging: boolean
  isAnimatingPan: boolean
  sizeVersion: number

  // Refs exposed for advanced usage
  containerRef: React.RefObject<HTMLDivElement | null>
  nodeSizesRef: React.RefObject<Map<Id, NodeSize>>
  zoomRef: React.RefObject<number>
}

export interface GraphCanvasActions<Id = string> {
  addNode: (id: Id, position: NodePosition) => void
  removeNode: (id: Id) => void
  setNodes: React.Dispatch<React.SetStateAction<Map<Id, NodePosition>>>
  setEdges: React.Dispatch<React.SetStateAction<CanvasEdge<Id>[]>>
  selectNode: (id: Id | null) => void
  setGraphRoot: (id: Id | null) => void
  setZoom: (zoom: number) => void
  expandNode: (fromId: Id, toId: Id, direction: { dx: number; dy: number }, edgeType?: string) => void
  animatePanTo: (target: { x: number; y: number }) => void
  panToNode: (id: Id) => void
  resetView: () => void
  clearGraph: () => void
  observeNodeRef: (nodeId: Id) => (el: HTMLDivElement | null) => void
}

export interface GraphCanvasHandlers {
  onMouseDown: (e: React.MouseEvent) => void
  onMouseMove: (e: React.MouseEvent) => void
  onMouseUp: (e: React.MouseEvent) => void
  onWheel: (e: React.WheelEvent) => void
}

export type GraphCanvas<Id = string> = GraphCanvasState<Id> & GraphCanvasActions<Id> & {
  handlers: GraphCanvasHandlers
}

// ─── Component Props ─────────────────────────────────────

export interface RenderNodeProps<Id = string> {
  id: Id
  isSelected: boolean
  isRoot: boolean
}

export interface GraphCanvasProps<Id = string> {
  canvas: GraphCanvas<Id>
  renderNode: (props: RenderNodeProps<Id>) => ReactNode
  theme?: Partial<GraphCanvasTheme>
  showDefaultControls?: boolean
  renderControls?: () => ReactNode
  renderEmpty?: () => ReactNode
  className?: string
  style?: React.CSSProperties
}

// ─── Theme ───────────────────────────────────────────────

export interface GraphCanvasTheme {
  background: string
  backgroundStars: boolean
  vignette: boolean
  edgeColors: Record<string, string>
  edgeWidth: number
  edgeGlowWidth: number
  edgeOpacity: number
  edgeGlowOpacity: number
  edgeDashArray: string
  arrowSize: { width: number; height: number }
  arrowOpacity: number
  panTransitionMs: number
  panTransitionEasing: string
  controlsBackground: string
  controlsBorder: string
  controlsText: string
  controlsHoverBackground: string
  controlsHoverText: string
  cursorDefault: string
  cursorDragging: string
}

// ─── Edge Components ─────────────────────────────────────

export interface EdgeLayerProps<Id = string> {
  edges: CanvasEdge<Id>[]
  nodes: Map<Id, NodePosition>
  nodeSizes: Map<Id, NodeSize>
  edgeColors: Record<string, string>
  arrowSize?: { width: number; height: number }
  arrowOpacity?: number
  edgeWidth?: number
  edgeGlowWidth?: number
  edgeOpacity?: number
  edgeGlowOpacity?: number
  edgeDashArray?: string
  sizeVersion?: number
}

export interface EdgePathProps {
  from: NodePosition
  to: NodePosition
  type: string
  fromSize: NodeSize | null
  toSize: NodeSize | null
  color: string
  edgeWidth?: number
  glowWidth?: number
  edgeOpacity?: number
  glowOpacity?: number
  dashArray?: string
  arrowId: string
}

export interface ConnectorLineProps {
  direction: 'vertical' | 'horizontal'
  color: string
  length?: number
}

export interface DefaultControlsProps {
  zoom: number
  nodeCount: number
  minZoom: number
  maxZoom: number
  onZoomIn: () => void
  onZoomOut: () => void
  onResetView: () => void
  onClearGraph: () => void
  theme?: Partial<GraphCanvasTheme>
}
