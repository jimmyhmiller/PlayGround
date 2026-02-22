import { useState, useCallback, useRef } from 'react'
import type {
  NodePosition,
  CanvasEdge,
  GraphCanvas,
  GraphCanvasHandlers,
  UseGraphCanvasOptions,
} from '../types'
import { computeExpandPosition } from '../layout/expand'
import { useResizeObserver } from './use-resize-observer'
import { useAnimatedPan } from './use-animated-pan'
import { useKeyboard } from './use-keyboard'

const DEFAULT_MIN_ZOOM = 0.15
const DEFAULT_MAX_ZOOM = 3
const DEFAULT_SPACING = 600
const DEFAULT_COLLISION_THRESHOLD = 300

/**
 * Core hook that manages all graph canvas state: nodes, edges, pan, zoom, drag, selection.
 *
 * Generic `Id` parameter defaults to `string` but supports `number` or any type.
 * Returns state, actions, and mouse/wheel handlers to attach to the canvas container.
 */
export function useGraphCanvas<Id = string>(
  options: UseGraphCanvasOptions = {}
): GraphCanvas<Id> {
  const {
    minZoom = DEFAULT_MIN_ZOOM,
    maxZoom = DEFAULT_MAX_ZOOM,
    spacing = DEFAULT_SPACING,
    collisionThreshold = DEFAULT_COLLISION_THRESHOLD,
    requireModifierKey = true,
  } = options

  // ─── Core State ────────────────────────────────────────

  const [nodes, setNodes] = useState<Map<Id, NodePosition>>(new Map())
  const [edges, setEdges] = useState<CanvasEdge<Id>[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<Id | null>(null)
  const [graphRoot, setGraphRoot] = useState<Id | null>(null)

  // Pan/zoom
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1)
  zoomRef.current = zoom
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Stale-closure-safe refs
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes
  const requireModRef = useRef(requireModifierKey)
  requireModRef.current = requireModifierKey

  // Node size measurement
  const { nodeSizesRef, sizeVersion, observeNodeRef } = useResizeObserver<Id>()

  // Animated pan
  const { isAnimatingPan, animatePanTo, cancelAnimation } = useAnimatedPan(
    setPan,
    400
  )

  // Drag tracking
  const dragRef = useRef<{
    type: 'pan' | 'node'
    nodeId?: Id
    startMouse: { x: number; y: number }
    startValue: { x: number; y: number }
    moved: boolean
  } | null>(null)

  // ─── Actions ───────────────────────────────────────────

  const addNode = useCallback((id: Id, position: NodePosition) => {
    setNodes((prev) => new Map(prev).set(id, position))
  }, [])

  const removeNode = useCallback(
    (nodeId: Id) => {
      if (nodeId === graphRoot) return
      setNodes((prev) => {
        const next = new Map(prev)
        next.delete(nodeId)
        return next
      })
      setEdges((prev) =>
        prev.filter((e) => e.fromId !== nodeId && e.toId !== nodeId)
      )
      nodeSizesRef.current.delete(nodeId)
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(graphRoot)
      }
    },
    [graphRoot, selectedNodeId, nodeSizesRef]
  )

  const selectNode = useCallback((nodeId: Id | null) => {
    setSelectedNodeId(nodeId)
  }, [])

  const expandNode = useCallback(
    (
      fromId: Id,
      toId: Id,
      direction: { dx: number; dy: number },
      edgeType: string = 'default'
    ) => {
      const currentNodes = nodesRef.current
      if (currentNodes.has(toId)) {
        const pos = currentNodes.get(toId)!
        animatePanTo({ x: -pos.x * zoomRef.current, y: -pos.y * zoomRef.current })
        setSelectedNodeId(toId)
        return
      }
      const fromPos = currentNodes.get(fromId)
      if (!fromPos) return

      const newPos = computeExpandPosition(
        fromPos,
        direction,
        currentNodes.values(),
        spacing,
        collisionThreshold
      )

      setNodes((prev) => new Map(prev).set(toId, newPos))
      setEdges((prev) => [...prev, { fromId, toId, type: edgeType }])
      setSelectedNodeId(toId)
      animatePanTo({
        x: -newPos.x * zoomRef.current,
        y: -newPos.y * zoomRef.current,
      })
    },
    [animatePanTo, spacing, collisionThreshold]
  )

  const panToNode = useCallback(
    (id: Id) => {
      const pos = nodesRef.current.get(id)
      if (pos) {
        animatePanTo({ x: -pos.x * zoomRef.current, y: -pos.y * zoomRef.current })
      }
    },
    [animatePanTo]
  )

  const resetView = useCallback(() => {
    setPan({ x: 0, y: 0 })
    setZoom(1)
  }, [])

  const clearGraph = useCallback(() => {
    if (graphRoot != null) {
      setNodes(new Map([[graphRoot, { x: 0, y: 0 }]]))
      setEdges([])
      setSelectedNodeId(graphRoot)
      nodeSizesRef.current.clear()
      setPan({ x: 0, y: 0 })
      setZoom(1)
    }
  }, [graphRoot, nodeSizesRef])

  // Delete/Backspace key handling
  useKeyboard(selectedNodeId, graphRoot, removeNode)

  // ─── Mouse Handlers ────────────────────────────────────

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return
      const target = e.target as HTMLElement
      if (target.closest('button, a, pre, input, [role="button"]')) return

      const nodeEl = target.closest('[data-drag-handle]') as HTMLElement | null
      if (nodeEl) {
        const nodeIdStr = nodeEl.dataset.dragHandle!
        // Find the matching node by stringified ID comparison
        let nodeId: Id | undefined
        for (const key of nodesRef.current.keys()) {
          if (String(key) === nodeIdStr) {
            nodeId = key
            break
          }
        }
        if (nodeId !== undefined) {
          const pos = nodesRef.current.get(nodeId)
          if (pos) {
            setSelectedNodeId(nodeId)
            if (!requireModRef.current || e.metaKey) {
              dragRef.current = {
                type: 'node',
                nodeId,
                startMouse: { x: e.clientX, y: e.clientY },
                startValue: { ...pos },
                moved: false,
              }
              setIsDragging(true)
              e.preventDefault()
            }
            return
          }
        }
      }

      if (requireModRef.current && !e.metaKey) return
      cancelAnimation()
      dragRef.current = {
        type: 'pan',
        startMouse: { x: e.clientX, y: e.clientY },
        startValue: { ...pan },
        moved: false,
      }
      setIsDragging(true)
      e.preventDefault()
    },
    [pan, cancelAnimation]
  )

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.startMouse.x
    const dy = e.clientY - dragRef.current.startMouse.y
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragRef.current.moved = true

    if (dragRef.current.type === 'pan') {
      setPan({
        x: dragRef.current.startValue.x + dx,
        y: dragRef.current.startValue.y + dy,
      })
    } else if (dragRef.current.nodeId != null) {
      const nid = dragRef.current.nodeId
      const sv = dragRef.current.startValue
      const z = zoomRef.current
      setNodes((prev) => {
        const next = new Map(prev)
        next.set(nid, { x: sv.x + dx / z, y: sv.y + dy / z })
        return next
      })
    }
  }, [])

  const handleMouseUp = useCallback(() => {
    dragRef.current = null
    setIsDragging(false)
  }, [])

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!requireModRef.current || e.metaKey) {
        // Scroll to pan canvas (Cmd+scroll when modifier required, plain scroll otherwise)
        e.preventDefault()
        cancelAnimation()
        setPan((p) => ({ x: p.x - e.deltaX, y: p.y - e.deltaY }))
      } else if (e.ctrlKey) {
        // Trackpad pinch — zoom toward cursor
        e.preventDefault()
        cancelAnimation()
        const container = containerRef.current
        if (!container) return
        const rect = container.getBoundingClientRect()
        const cx = e.clientX - rect.left - rect.width / 2
        const cy = e.clientY - rect.top - rect.height / 2
        const delta = e.deltaY > 0 ? 0.95 : 1.05
        const currentZoom = zoomRef.current
        const newZoom = Math.min(
          maxZoom,
          Math.max(minZoom, currentZoom * delta)
        )
        const scale = newZoom / currentZoom
        zoomRef.current = newZoom
        setPan((p) => ({
          x: cx - scale * (cx - p.x),
          y: cy - scale * (cy - p.y),
        }))
        setZoom(newZoom)
      }
    },
    [cancelAnimation, minZoom, maxZoom]
  )

  const handlers: GraphCanvasHandlers = {
    onMouseDown: handleMouseDown,
    onMouseMove: handleMouseMove,
    onMouseUp: handleMouseUp,
    onWheel: handleWheel,
  }

  return {
    // State
    nodes,
    edges,
    selectedNodeId,
    graphRoot,
    pan,
    zoom,
    isDragging,
    isAnimatingPan,
    sizeVersion,
    containerRef,
    nodeSizesRef,
    zoomRef,

    // Actions
    addNode,
    removeNode,
    setNodes,
    setEdges,
    selectNode,
    setGraphRoot,
    setZoom,
    expandNode,
    animatePanTo,
    panToNode,
    resetView,
    clearGraph,
    observeNodeRef,

    // Handlers
    handlers,
  }
}
