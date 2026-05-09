'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useTaskData } from '@/hooks/use-task-data'
import { useControlStatus } from '@/hooks/use-control-status'
import { controlApi, CellDetail, CellInfo } from '@/lib/control-api'
import { middleTruncate } from '@/lib/utils'
import { Pause, Play, SkipForward, Maximize2, Minimize2 } from 'lucide-react'
import { CellContentRenderer } from './cell-renderer'
import dagre from '@dagrejs/dagre'

// ─── Types ──────────────────────────────────────────────────

interface Props {
  taskId: number | null
  onSelectTask: (taskId: number) => void
}

interface NodePosition { x: number; y: number }
interface NodeSize { w: number; h: number }

interface CanvasEdge {
  fromId: number
  toId: number
  type: 'child' | 'dep' | 'dependent'
}

// ─── Constants ──────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  scheduled: '#fbbf24',
  in_progress: '#22d3ee',
  completed: '#34d399',
  dirty: '#fb7185',
  created: '#71717a',
}

const PORT_COLORS = {
  dependents: { accent: '#fbbf24', glow: 'rgba(251,191,36,0.15)', hoverBg: 'rgba(251,191,36,0.08)' },
  children: { accent: '#60a5fa', glow: 'rgba(96,165,250,0.15)', hoverBg: 'rgba(96,165,250,0.08)' },
  deps: { accent: '#c084fc', glow: 'rgba(192,132,252,0.15)', hoverBg: 'rgba(192,132,252,0.08)' },
  cells: { accent: '#34d399', glow: 'rgba(52,211,153,0.15)', hoverBg: 'rgba(52,211,153,0.08)' },
}

const EDGE_COLORS: Record<string, string> = {
  child: '#60a5fa',
  dep: '#c084fc',
  dependent: '#fbbf24',
}

const MIN_ZOOM = 0.15
const MAX_ZOOM = 3
const SPACING = 600
const COLLISION_THRESHOLD = 300
// Fallback half-sizes if ResizeObserver hasn't reported yet
const FALLBACK_HALF_H = 100
const FALLBACK_HALF_W = 160

// DAG view constants
const DAG_NODE_W = 220
const DAG_NODE_H = 30
const DAG_NODESEP = 16
const DAG_RANKSEP = 80

function stateColor(state: string) { return STATE_COLORS[state] || '#71717a' }

function stateClass(state: string) {
  if (state === 'dirty') return 'state-dirty'
  if (state === 'in_progress') return 'state-in_progress'
  if (state === 'completed') return 'state-completed'
  if (state === 'scheduled') return 'state-scheduled'
  return ''
}

// ─── Main Component ─────────────────────────────────────────

export function SpaceExplorer({ taskId, onSelectTask }: Props) {
  const [nodes, setNodes] = useState<Map<number, NodePosition>>(new Map())
  const [edges, setEdges] = useState<CanvasEdge[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)
  const [graphRoot, setGraphRoot] = useState<number | null>(null) // undeletable origin node
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showOnlyOpened, setShowOnlyOpened] = useState(false)
  const [autoAddTasks, setAutoAddTasks] = useState(false)
  const [viewMode, setViewMode] = useState<'canvas' | 'dag'>('canvas')
  const { pending } = useControlStatus()
  const seenTasksRef = useRef<Set<number>>(new Set())

  // Pan/zoom
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1) // updated eagerly for rapid pinch events
  zoomRef.current = zoom // sync from state (buttons, reset, etc.)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Node size measurement via ResizeObserver
  const nodeSizesRef = useRef<Map<number, NodeSize>>(new Map())
  const [sizeVersion, setSizeVersion] = useState(0)
  const observerRef = useRef<ResizeObserver | null>(null)

  useEffect(() => {
    observerRef.current = new ResizeObserver((entries) => {
      let changed = false
      for (const entry of entries) {
        const el = entry.target as HTMLElement
        const idStr = el.dataset.nodeSize
        if (!idStr) continue
        const nodeId = parseInt(idStr, 10)
        const w = Math.round(entry.contentRect.width)
        const h = Math.round(entry.contentRect.height)
        const prev = nodeSizesRef.current.get(nodeId)
        if (!prev || prev.w !== w || prev.h !== h) {
          nodeSizesRef.current.set(nodeId, { w, h })
          changed = true
        }
      }
      if (changed) setSizeVersion(v => v + 1)
    })
    return () => observerRef.current?.disconnect()
  }, [])

  const observeNodeRef = useCallback((nodeId: number) => (el: HTMLDivElement | null) => {
    if (el && observerRef.current) {
      el.dataset.nodeSize = String(nodeId)
      observerRef.current.observe(el)
    }
  }, [])

  // Animated pan transition — enabled when programmatically centering, disabled during drag/scroll
  const [isAnimatingPan, setIsAnimatingPan] = useState(false)
  const animatingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const animatePanTo = useCallback((target: { x: number; y: number }) => {
    // Cancel any pending animation end timer
    if (animatingTimerRef.current) clearTimeout(animatingTimerRef.current)
    setIsAnimatingPan(true)
    setPan(target)
    // Turn off transition after it completes (400ms matches the CSS transition duration)
    animatingTimerRef.current = setTimeout(() => setIsAnimatingPan(false), 420)
  }, [])

  // Drag tracking
  const dragRef = useRef<{
    type: 'pan' | 'node'
    nodeId?: number
    startMouse: { x: number; y: number }
    startValue: { x: number; y: number }
    moved: boolean
  } | null>(null)

  const nodesRef = useRef(nodes)
  nodesRef.current = nodes

  // Stable ref to addTaskToCanvas so the effect doesn't re-fire on pan/zoom changes
  const addTaskToCanvasRef = useRef<(id: number) => void>(() => {})

  // Initialize when root taskId changes — don't reset if task is already on canvas
  useEffect(() => {
    if (taskId != null) {
      if (nodesRef.current.has(taskId)) {
        // Already on canvas — just select it
        setSelectedNodeId(taskId)
      } else if (nodesRef.current.size === 0) {
        // Empty canvas — fresh start
        setNodes(new Map([[taskId, { x: 0, y: 0 }]]))
        setEdges([])
        setPan({ x: 0, y: 0 })
        setZoom(1)
        nodeSizesRef.current.clear()
        setGraphRoot(taskId)
        setSelectedNodeId(taskId)
      } else {
        // Canvas has nodes — add this task connected to existing nodes
        addTaskToCanvasRef.current(taskId)
      }
    } else {
      setNodes(new Map())
      setEdges([])
      setSelectedNodeId(null)
      setGraphRoot(null)
      nodeSizesRef.current.clear()
    }
  }, [taskId])

  const existingNodeIds = useMemo(() => new Set(nodes.keys()), [nodes])

  // Auto-add tasks when they appear in pending queue
  useEffect(() => {
    if (!autoAddTasks || nodes.size === 0) return

    for (const task of pending) {
      // Only add if we haven't seen it before and it's not already on canvas
      if (!seenTasksRef.current.has(task.task_id) && !nodes.has(task.task_id)) {
        seenTasksRef.current.add(task.task_id)
        addTaskToCanvasRef.current(task.task_id)
      }
    }
  }, [pending, autoAddTasks, nodes])

  const expandNode = useCallback((fromId: number, toId: number, type: CanvasEdge['type']) => {
    if (nodes.has(toId)) {
      // Already on canvas — animate to it
      const pos = nodes.get(toId)!
      animatePanTo({ x: -pos.x * zoom, y: -pos.y * zoom })
      setSelectedNodeId(toId)
      onSelectTask(toId)
      return
    }
    const fromPos = nodes.get(fromId)
    if (!fromPos) return

    let x = fromPos.x, y = fromPos.y
    switch (type) {
      case 'child': y += SPACING; break
      case 'dep': x -= SPACING; break
      case 'dependent': y -= SPACING; break
    }

    let attempts = 0
    const occupied = (tx: number, ty: number) => {
      for (const pos of nodes.values()) {
        if (Math.abs(pos.x - tx) < COLLISION_THRESHOLD && Math.abs(pos.y - ty) < COLLISION_THRESHOLD) return true
      }
      return false
    }
    while (occupied(x, y) && attempts < 20) {
      if (type === 'child' || type === 'dependent') x += SPACING * 0.6
      else y += SPACING * 0.6
      attempts++
    }

    setNodes(prev => new Map(prev).set(toId, { x, y }))
    setEdges(prev => [...prev, { fromId, toId, type }])
    setSelectedNodeId(toId)
    onSelectTask(toId)
    // Animate viewport to the new node
    animatePanTo({ x: -x * zoom, y: -y * zoom })
  }, [nodes, onSelectTask, zoom, animatePanTo])

  const removeNode = useCallback((nodeId: number) => {
    if (nodeId === graphRoot) return // can't remove graph origin
    setNodes(prev => { const next = new Map(prev); next.delete(nodeId); return next })
    setEdges(prev => prev.filter(e => e.fromId !== nodeId && e.toId !== nodeId))
    nodeSizesRef.current.delete(nodeId)
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(graphRoot)
      if (graphRoot != null) onSelectTask(graphRoot)
    }
  }, [graphRoot, selectedNodeId, onSelectTask])

  const selectNode = useCallback((nodeId: number) => {
    setSelectedNodeId(nodeId)
    onSelectTask(nodeId)
  }, [onSelectTask])

  // Add a task from the sidebar queue directly onto the canvas
  // Tries to find an existing related node (child, dep, dependent) to connect to
  const addTaskToCanvas = useCallback(async (addId: number) => {
    if (nodesRef.current.has(addId)) {
      const pos = nodesRef.current.get(addId)!
      animatePanTo({ x: -pos.x * zoom, y: -pos.y * zoom })
      setSelectedNodeId(addId)
      onSelectTask(addId)
      return
    }

    const occupied = (tx: number, ty: number) => {
      for (const pos of nodesRef.current.values()) {
        if (Math.abs(pos.x - tx) < COLLISION_THRESHOLD && Math.abs(pos.y - ty) < COLLISION_THRESHOLD) return true
      }
      return false
    }

    // Find ALL related nodes already on canvas and collect edges
    const newEdges: CanvasEdge[] = []
    // Use the first anchor for positioning
    let anchorId: number | null = null
    let offsetX = 0, offsetY = 0

    try {
      const [children, deps] = await Promise.all([
        controlApi.liveChildren(addId).catch(() => []),
        controlApi.liveDeps(addId).catch(() => null),
      ])

      const currentNodes = nodesRef.current

      // Children: newTask is parent of existing nodes
      for (const child of children) {
        if (currentNodes.has(child.task_id)) {
          newEdges.push({ fromId: addId, toId: child.task_id, type: 'child' })
          if (!anchorId) { anchorId = child.task_id; offsetY = -SPACING }
        }
      }

      if (deps) {
        // Deps: newTask reads from existing nodes
        const allDeps = [...deps.output_deps.map(([id]) => id), ...deps.cell_deps.map(([id]) => id)]
        const seen = new Set<number>()
        for (const depId of allDeps) {
          if (currentNodes.has(depId) && !seen.has(depId)) {
            seen.add(depId)
            newEdges.push({ fromId: addId, toId: depId, type: 'dep' })
            if (!anchorId) { anchorId = depId; offsetX = SPACING }
          }
        }

        // Dependents: existing nodes read from newTask
        for (const [depId] of deps.dependents) {
          if (currentNodes.has(depId) && !seen.has(depId)) {
            seen.add(depId)
            newEdges.push({ fromId: depId, toId: addId, type: 'dep' })
            if (!anchorId) { anchorId = depId; offsetX = -SPACING }
          }
        }
      }
    } catch { /* fall through to unconnected placement */ }

    if (anchorId != null && newEdges.length > 0) {
      const anchorPos = nodesRef.current.get(anchorId)!
      let x = anchorPos.x + offsetX
      let y = anchorPos.y + offsetY
      let attempts = 0
      while (occupied(x, y) && attempts < 20) {
        if (offsetY !== 0) x += SPACING * 0.6
        else y += SPACING * 0.6
        attempts++
      }
      setNodes(prev => new Map(prev).set(addId, { x, y }))
      setEdges(prev => [...prev, ...newEdges])
      setSelectedNodeId(addId)
      onSelectTask(addId)
      animatePanTo({ x: -x * zoom, y: -y * zoom })
    } else {
      // Fallback: place near center of current view
      const viewCenterX = -pan.x / zoom
      const viewCenterY = -pan.y / zoom
      let x = viewCenterX + 300, y = viewCenterY
      let attempts = 0
      while (occupied(x, y) && attempts < 20) { y += SPACING * 0.5; attempts++ }
      setNodes(prev => new Map(prev).set(addId, { x, y }))
      setSelectedNodeId(addId)
      onSelectTask(addId)
      animatePanTo({ x: -x * zoom, y: -y * zoom })
    }
  }, [pan, zoom, onSelectTask, animatePanTo])
  addTaskToCanvasRef.current = addTaskToCanvas

  // ─── Mouse handlers ──────────────────────────────────

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    const target = e.target as HTMLElement
    if (target.closest('button, a, pre, input, [role="button"]')) return

    // Cmd+drag to move nodes or pan canvas; plain click selects node
    const nodeEl = target.closest('[data-drag-handle]') as HTMLElement | null
    if (nodeEl) {
      const nodeId = parseInt(nodeEl.dataset.dragHandle!, 10)
      const pos = nodes.get(nodeId)
      if (pos) {
        selectNode(nodeId)
        if (e.metaKey) {
          dragRef.current = { type: 'node', nodeId, startMouse: { x: e.clientX, y: e.clientY }, startValue: { ...pos }, moved: false }
          setIsDragging(true)
          e.preventDefault()
        }
        return
      }
    }

    // Cmd+drag on background = pan canvas
    if (!e.metaKey) return
    setIsAnimatingPan(false)
    if (animatingTimerRef.current) { clearTimeout(animatingTimerRef.current); animatingTimerRef.current = null }
    dragRef.current = { type: 'pan', startMouse: { x: e.clientX, y: e.clientY }, startValue: { ...pan }, moved: false }
    setIsDragging(true)
    e.preventDefault()
  }, [nodes, pan, selectNode])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.startMouse.x
    const dy = e.clientY - dragRef.current.startMouse.y
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragRef.current.moved = true

    if (dragRef.current.type === 'pan') {
      setPan({ x: dragRef.current.startValue.x + dx, y: dragRef.current.startValue.y + dy })
    } else if (dragRef.current.nodeId != null) {
      const nid = dragRef.current.nodeId
      const sv = dragRef.current.startValue
      const z = zoomRef.current
      setNodes(prev => {
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

  // Cmd+scroll = pan canvas, Ctrl+scroll (trackpad pinch) = zoom
  // Regular scroll passes through to child elements.
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.metaKey) {
      // Cmd+scroll — pan canvas
      e.preventDefault()
      setIsAnimatingPan(false)
      if (animatingTimerRef.current) { clearTimeout(animatingTimerRef.current); animatingTimerRef.current = null }
      setPan(p => ({ x: p.x - e.deltaX, y: p.y - e.deltaY }))
    } else if (e.ctrlKey) {
      // Trackpad pinch (browser synthesizes ctrlKey) — zoom toward cursor
      e.preventDefault()
      setIsAnimatingPan(false)
      if (animatingTimerRef.current) { clearTimeout(animatingTimerRef.current); animatingTimerRef.current = null }
      const container = containerRef.current
      if (!container) return
      const rect = container.getBoundingClientRect()
      const cx = e.clientX - rect.left - rect.width / 2
      const cy = e.clientY - rect.top - rect.height / 2
      const delta = e.deltaY > 0 ? 0.95 : 1.05
      // Read from ref so rapid pinch events accumulate correctly
      const currentZoom = zoomRef.current
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, currentZoom * delta))
      const scale = newZoom / currentZoom
      zoomRef.current = newZoom // update immediately for next event
      setPan(p => ({ x: cx - scale * (cx - p.x), y: cy - scale * (cy - p.y) }))
      setZoom(newZoom)
    }
    // else: regular scroll passes through
  }, [])

  const resetView = useCallback(() => { setPan({ x: 0, y: 0 }); setZoom(1) }, [])

  const clearGraph = useCallback(() => {
    if (graphRoot != null) {
      setNodes(new Map([[graphRoot, { x: 0, y: 0 }]]))
      setEdges([])
      setSelectedNodeId(graphRoot)
      onSelectTask(graphRoot)
      nodeSizesRef.current.clear()
      setPan({ x: 0, y: 0 })
      setZoom(1)
    }
  }, [graphRoot, onSelectTask])

  // Find all missing edges between existing nodes on canvas
  const [reconnecting, setReconnecting] = useState(false)
  const reconnectAll = useCallback(async () => {
    const nodeIds = Array.from(nodesRef.current.keys())
    if (nodeIds.length < 2) return
    setReconnecting(true)
    try {
      const nodeSet = new Set(nodeIds)
      const newEdges: CanvasEdge[] = []
      // Fetch children and deps for every node in parallel
      const results = await Promise.all(
        nodeIds.map(async (id) => {
          const [children, deps] = await Promise.all([
            controlApi.liveChildren(id).catch(() => []),
            controlApi.liveDeps(id).catch(() => null),
          ])
          return { id, children, deps }
        })
      )
      // Build a set of existing edges for dedup
      const existingEdgeKeys = new Set(
        edges.map(e => `${e.fromId}-${e.toId}-${e.type}`)
      )
      const addEdge = (edge: CanvasEdge) => {
        const key = `${edge.fromId}-${edge.toId}-${edge.type}`
        if (!existingEdgeKeys.has(key)) {
          existingEdgeKeys.add(key)
          newEdges.push(edge)
        }
      }
      for (const { id, children, deps } of results) {
        for (const child of children) {
          if (nodeSet.has(child.task_id)) {
            addEdge({ fromId: id, toId: child.task_id, type: 'child' })
          }
        }
        if (deps) {
          for (const [depId] of deps.output_deps) {
            if (nodeSet.has(depId)) addEdge({ fromId: id, toId: depId, type: 'dep' })
          }
          for (const [depId] of deps.cell_deps) {
            if (nodeSet.has(depId)) addEdge({ fromId: id, toId: depId, type: 'dep' })
          }
        }
      }
      if (newEdges.length > 0) {
        setEdges(prev => [...prev, ...newEdges])
      }
    } catch (e) {
      console.error('reconnectAll failed:', e)
    } finally {
      setReconnecting(false)
    }
  }, [edges])

  // Native fullscreen API
  const fullscreenRef = useRef<HTMLDivElement>(null)

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      fullscreenRef.current?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }, [])

  // Delete/Backspace removes selected node — window-level so focus isn't required
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const tag = (e.target as HTMLElement)?.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA') return
        if (selectedNodeId != null && selectedNodeId !== graphRoot) {
          e.preventDefault()
          removeNode(selectedNodeId)
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedNodeId, graphRoot, removeNode])

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  if (taskId == null) {
    return (
      <div className="space-bg flex h-full items-center justify-center">
        <div className="relative z-10 text-center">
          <div className="text-sm text-zinc-500">Select a task to explore</div>
          <div className="mt-1 text-[10px] text-zinc-600">Click a task in the event log, search, or pending queue</div>
        </div>
      </div>
    )
  }

  // DAG layout computation
  const dagPositions = useMemo(() => {
    if (viewMode !== 'dag' || nodes.size === 0) return null

    const g = new dagre.graphlib.Graph()
    g.setDefaultEdgeLabel(() => ({}))
    g.setGraph({ rankdir: 'LR', nodesep: DAG_NODESEP, ranksep: DAG_RANKSEP })

    const nodeIds = Array.from(nodes.keys())
    for (const id of nodeIds) {
      g.setNode(String(id), { width: DAG_NODE_W, height: DAG_NODE_H })
    }
    for (const edge of edges) {
      if (nodes.has(edge.fromId) && nodes.has(edge.toId)) {
        if (edge.type === 'dep') {
          // Dep edges stored as depender→dependency; reverse so dependency (LEFT) → depender (RIGHT)
          g.setEdge(String(edge.toId), String(edge.fromId))
        } else {
          g.setEdge(String(edge.fromId), String(edge.toId))
        }
      }
    }

    dagre.layout(g)

    // Offset so graphRoot is at (0,0)
    const rootPos = graphRoot != null ? g.node(String(graphRoot)) : null
    const offsetX = rootPos ? rootPos.x : 0
    const offsetY = rootPos ? rootPos.y : 0

    const positions = new Map<number, { x: number; y: number }>()
    for (const id of nodeIds) {
      const node = g.node(String(id))
      if (node) {
        positions.set(id, { x: node.x - offsetX, y: node.y - offsetY })
      }
    }

    return positions
  }, [viewMode, nodes, edges, graphRoot])

  const nodeList = Array.from(nodes.entries())

  const canvas = (
    <div ref={fullscreenRef} className="space-bg relative flex h-full overflow-hidden outline-none" tabIndex={0}>
      {/* Fullscreen sidebar */}
      {isFullscreen && (
        <ExplorerSidebar
          existingNodeIds={existingNodeIds}
          onAddTask={addTaskToCanvas}
        />
      )}

      <div className="relative flex-1 overflow-hidden">
        {/* Controls overlay */}
        <div className="absolute bottom-2 right-3 z-20 flex items-center gap-2">
          <span className="font-mono text-[9px] text-zinc-600">{nodes.size} node{nodes.size !== 1 ? 's' : ''}</span>
          <button className="rounded border border-zinc-700 bg-zinc-900/80 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300" onClick={clearGraph}>Clear</button>
          <button className="rounded border border-zinc-700 bg-zinc-900/80 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300" onClick={resetView}>Reset view</button>
          <button
            className="rounded border border-zinc-700 bg-zinc-900/80 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
            onClick={reconnectAll}
            disabled={reconnecting || nodes.size < 2}
          >
            {reconnecting ? 'Connecting...' : 'Reconnect'}
          </button>
          <button
            className={`rounded border px-2 py-1 text-[10px] ${showOnlyOpened ? 'border-blue-500 bg-blue-500/20 text-blue-300' : 'border-zinc-700 bg-zinc-900/80 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300'}`}
            onClick={() => setShowOnlyOpened(!showOnlyOpened)}
          >
            {showOnlyOpened ? 'Show all' : 'Show only opened'}
          </button>
          <button
            className={`rounded border px-2 py-1 text-[10px] ${autoAddTasks ? 'border-green-500 bg-green-500/20 text-green-300' : 'border-zinc-700 bg-zinc-900/80 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300'}`}
            onClick={() => setAutoAddTasks(!autoAddTasks)}
            title="Automatically add tasks to canvas as they are queued"
          >
            Auto-add {autoAddTasks ? 'ON' : 'OFF'}
          </button>
          <button
            className={`rounded border px-2 py-1 text-[10px] ${viewMode === 'dag' ? 'border-cyan-500 bg-cyan-500/20 text-cyan-300' : 'border-zinc-700 bg-zinc-900/80 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300'}`}
            onClick={() => {
              const next = viewMode === 'canvas' ? 'dag' : 'canvas'
              setViewMode(next)
              if (next === 'dag') { setPan({ x: 0, y: 0 }); setZoom(1) }
            }}
            title="Toggle structured DAG layout"
          >
            {viewMode === 'dag' ? 'Free layout' : 'DAG view'}
          </button>
          <button
            className="rounded border border-zinc-700 bg-zinc-900/80 p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
          <button className="rounded border border-zinc-700 bg-zinc-900/80 px-1.5 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300" onClick={() => setZoom(z => Math.max(MIN_ZOOM, z * 0.8))}>−</button>
          <span className="font-mono text-[9px] text-zinc-600">{Math.round(zoom * 100)}%</span>
          <button className="rounded border border-zinc-700 bg-zinc-900/80 px-1.5 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300" onClick={() => setZoom(z => Math.min(MAX_ZOOM, z * 1.25))}>+</button>
        </div>

        {/* Canvas */}
        <div
          ref={containerRef}
          className="h-full w-full"
          style={{ cursor: isDragging ? 'grabbing' : 'default' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
          {/* World */}
          <div
            className="absolute left-1/2 top-1/2"
            style={{
              transform: `translate(calc(-50% + ${pan.x}px), calc(-50% + ${pan.y}px)) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: isAnimatingPan ? 'transform 400ms cubic-bezier(0.25, 0.46, 0.45, 0.94)' : 'none',
            }}
          >
            {viewMode === 'dag' && dagPositions ? (
              <>
                {/* DAG edge layer */}
                <svg style={{ position: 'absolute', left: 0, top: 0, width: 0, height: 0, overflow: 'visible', pointerEvents: 'none' }}>
                  <defs>
                    {Object.entries(EDGE_COLORS).map(([type, color]) => (
                      <marker key={`dag-${type}`} id={`dag-arrow-${type}`} viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="4" orient="auto">
                        <path d="M 0 0 L 10 3 L 0 6 Z" fill={color} opacity={0.6} />
                      </marker>
                    ))}
                  </defs>
                  {edges.map((edge, i) => {
                    const from = dagPositions.get(edge.fromId)
                    const to = dagPositions.get(edge.toId)
                    if (!from || !to) return null
                    return <DagEdgePath key={`${edge.fromId}-${edge.toId}-${i}`} from={from} to={to} type={edge.type} />
                  })}
                </svg>

                {/* DAG nodes */}
                {Array.from(dagPositions.entries()).map(([nodeId, pos]) => (
                  <div
                    key={nodeId}
                    className="absolute"
                    style={{ left: pos.x, top: pos.y, transform: 'translate(-50%, -50%)' }}
                  >
                    <DagNodeLabel
                      taskId={nodeId}
                      isRoot={nodeId === graphRoot}
                      isSelected={nodeId === selectedNodeId}
                      onClick={() => selectNode(nodeId)}
                    />
                  </div>
                ))}
              </>
            ) : (
              <>
                {/* SVG edge layer */}
                <svg style={{ position: 'absolute', left: 0, top: 0, width: 0, height: 0, overflow: 'visible', pointerEvents: 'none' }}>
                  <defs>
                    {Object.entries(EDGE_COLORS).map(([type, color]) => (
                      <marker key={type} id={`arrow-${type}`} viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="5" orient="auto">
                        <path d="M 0 0 L 10 3 L 0 6 Z" fill={color} opacity={0.7} />
                      </marker>
                    ))}
                  </defs>
                  {/* Use sizeVersion to re-render when node sizes change */}
                  {sizeVersion >= 0 && edges.map((edge, i) => {
                    const from = nodes.get(edge.fromId)
                    const to = nodes.get(edge.toId)
                    if (!from || !to) return null
                    const fromSize = nodeSizesRef.current.get(edge.fromId)
                    const toSize = nodeSizesRef.current.get(edge.toId)
                    return (
                      <EdgePath
                        key={`${edge.fromId}-${edge.toId}-${i}`}
                        from={from} to={to} type={edge.type}
                        fromSize={fromSize || null}
                        toSize={toSize || null}
                      />
                    )
                  })}
                </svg>

                {/* Nodes */}
                {nodeList.map(([nodeId, pos]) => (
                  <div
                    key={nodeId}
                    ref={observeNodeRef(nodeId)}
                    className="absolute"
                    style={{ left: pos.x, top: pos.y, transform: 'translate(-50%, -50%)' }}
                  >
                    <TaskNode
                      taskId={nodeId}
                      isRoot={nodeId === graphRoot}
                      isSelected={nodeId === selectedNodeId}
                      existingNodeIds={existingNodeIds}
                      showOnlyOpened={showOnlyOpened}
                      onExpand={(toId, type) => expandNode(nodeId, toId, type)}
                      onClose={() => removeNode(nodeId)}
                      onSelect={() => selectNode(nodeId)}
                    />
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )

  return canvas
}

// ─── ExplorerSidebar (fullscreen only) ──────────────────────

function ExplorerSidebar({ existingNodeIds, onAddTask }: {
  existingNodeIds: Set<number>
  onAddTask: (id: number) => void
}) {
  const { status, pending, pause, resume, step, stepTask } = useControlStatus()
  const [stepCount, setStepCount] = useState(1)

  return (
    <div className="relative z-10 flex w-60 shrink-0 flex-col border-r border-border bg-background text-foreground">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-1.5 border-b border-border px-2 py-2">
        {status?.paused ? (
          <button className="flex items-center gap-1 rounded bg-green-700 px-2 py-1 text-[10px] text-primary-foreground hover:bg-green-600" onClick={resume}>
            <Play className="h-3 w-3" /> Resume
          </button>
        ) : (
          <button className="flex items-center gap-1 rounded bg-red-700 px-2 py-1 text-[10px] text-primary-foreground hover:bg-red-600" onClick={pause}>
            <Pause className="h-3 w-3" /> Pause
          </button>
        )}
        <button
          className="flex items-center gap-1 rounded bg-secondary px-2 py-1 text-[10px] text-secondary-foreground hover:bg-secondary/80 disabled:opacity-40"
          onClick={() => step(stepCount)}
          disabled={!status?.paused}
        >
          <SkipForward className="h-3 w-3" /> Step
        </button>
        <input
          type="number" min={1} value={stepCount}
          onChange={e => setStepCount(Math.max(1, parseInt(e.target.value) || 1))}
          className="h-6 w-10 rounded border border-input bg-background text-center text-[10px] text-foreground"
        />
        <span className="text-xs text-muted-foreground">{status?.pending_count ?? '?'} pending</span>
      </div>

      {/* Queue */}
      <div className="flex items-center gap-1 border-b border-border px-2 py-1.5">
        <span className="text-[9px] font-bold uppercase tracking-wider text-foreground">Pending Queue</span>
        <span className="text-[9px] text-muted-foreground">({pending.length})</span>
      </div>
      <div className="flex-1 overflow-auto">
        {pending.length === 0 ? (
          <div className="px-2 py-4 text-center text-[10px] text-muted-foreground">No pending tasks</div>
        ) : (
          <div className="flex flex-col gap-px p-1">
            {pending.map(task => {
              const isOnCanvas = existingNodeIds.has(task.task_id)
              return (
                <div key={task.task_id} className="flex items-center gap-1 rounded px-1.5 py-1 hover:bg-accent">
                  <button
                    className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
                    onClick={() => onAddTask(task.task_id)}
                  >
                    <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${isOnCanvas ? 'bg-muted-foreground' : 'bg-amber-400'}`} />
                    <span className="font-mono text-[10px] text-foreground">#{task.task_id}</span>
                    <span className="min-w-0 truncate text-[9px] text-muted-foreground">{middleTruncate(task.name, 20)}</span>
                  </button>
                  {status?.paused && (
                    <button
                      className="shrink-0 rounded bg-secondary px-1.5 py-0.5 text-[9px] text-secondary-foreground hover:bg-secondary/80"
                      onClick={() => stepTask(task.task_id)}
                    >
                      release
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── TaskNode ───────────────────────────────────────────────

interface TaskNodeProps {
  taskId: number
  isRoot: boolean
  isSelected: boolean
  existingNodeIds: Set<number>
  showOnlyOpened: boolean
  onExpand: (toId: number, type: CanvasEdge['type']) => void
  onClose: () => void
  onSelect: () => void
}

function TaskNode({ taskId, isRoot, isSelected, existingNodeIds, showOnlyOpened, onExpand, onClose, onSelect }: TaskNodeProps) {
  const data = useTaskData(taskId)
  const [expandedPorts, setExpandedPorts] = useState<Set<string>>(new Set(['children', 'deps', 'dependents', 'cells']))
  const [expandedCell, setExpandedCell] = useState<number | null>(null)
  const [cellDetail, setCellDetail] = useState<CellDetail | null>(null)
  const [cellDetailLoading, setCellDetailLoading] = useState(false)

  const togglePort = (port: string) => {
    setExpandedPorts(prev => {
      const next = new Set(prev)
      if (next.has(port)) next.delete(port)
      else next.add(port)
      return next
    })
  }

  const handleExpandCell = (cellIndex: number) => {
    if (expandedCell === cellIndex) { setExpandedCell(null); setCellDetail(null); return }
    setExpandedCell(cellIndex)
    setCellDetailLoading(true)
    controlApi.cellDetail(taskId, cellIndex).then(d => { setCellDetail(d); setCellDetailLoading(false) }).catch(() => { setCellDetailLoading(false) })
  }

  const state = data.state
  const currentState = state?.state || 'created'
  const color = stateColor(currentState)

  const allDeps: { id: number; name: string; label?: string }[] = []
  if (data.deps) {
    for (const [id, name] of data.deps.output_deps) allDeps.push({ id, name, label: 'output' })
    for (const [id, name, idx] of data.deps.cell_deps) allDeps.push({ id, name, label: `cell ${idx}` })
  }
  const dependents = data.deps?.dependents || []

  // Filter items if showOnlyOpened is enabled
  const filteredDeps = showOnlyOpened ? allDeps.filter(d => existingNodeIds.has(d.id)) : allDeps
  const filteredDependents = showOnlyOpened ? dependents.filter(([id]) => existingNodeIds.has(id)) : dependents
  const filteredChildren = showOnlyOpened ? data.children.filter(c => existingNodeIds.has(c.task_id)) : data.children
  const filteredCells = showOnlyOpened && expandedCell !== null ? data.cells.filter(c => c.cell_index === expandedCell) : data.cells

  return (
    <div className="flex items-center gap-0">
      {/* Left: deps */}
      <ExpandablePort direction="left" label="deps" items={filteredDeps} colors={PORT_COLORS.deps}
        expanded={expandedPorts.has('deps')} onToggle={() => togglePort('deps')}
        onClickItem={(id) => onExpand(id, 'dep')} existingNodeIds={existingNodeIds} />

      <div className="flex flex-col items-center gap-0">
        {/* Top: dependents */}
        <ExpandablePort direction="top" label="dependents" items={filteredDependents.map(([id, name]) => ({ id, name }))}
          colors={PORT_COLORS.dependents} expanded={expandedPorts.has('dependents')}
          onToggle={() => togglePort('dependents')} onClickItem={(id) => onExpand(id, 'dependent')}
          existingNodeIds={existingNodeIds} />

        {/* Card */}
        <div
          data-drag-handle={taskId}
          className={`explorer-card-glow scan-overlay relative rounded-lg bg-zinc-950/80 px-5 py-3 ${stateClass(currentState)} ${data.stateChanged ? 'space-state-flash' : ''} ${isSelected ? 'ring-1 ring-blue-400/40' : ''}`}
          style={{ minWidth: 240, maxWidth: 340 }}
        >
          {!isRoot && (
            <button className="absolute right-1.5 top-1.5 flex h-4 w-4 items-center justify-center rounded text-[10px] text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
              onClick={(e) => { e.stopPropagation(); onClose() }}>&times;</button>
          )}
          <div className="mb-0.5 pr-4 font-mono text-xs font-semibold text-zinc-100">
            {state?.name ? middleTruncate(state.name, 45) : `Task #${taskId}`}
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-zinc-500">#{taskId}</span>
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
            <span className="text-[10px] text-zinc-400">{currentState}</span>
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-2 text-[9px] text-zinc-500">
            <span>{data.cells.length} cells</span><span>&middot;</span>
            <span>{data.children.length} children</span><span>&middot;</span>
            <span>{allDeps.length} deps</span><span>&middot;</span>
            <span>{dependents.length} dependents</span>
          </div>
          {state && (state.is_dirty || state.is_in_progress || state.is_stateful || state.is_immutable) && (
            <div className="mt-1 flex flex-wrap gap-1">
              {state.is_dirty && <FlagBadge label="dirty" color="#fb7185" />}
              {state.is_in_progress && <FlagBadge label="in_progress" color="#22d3ee" />}
              {state.is_stateful && <FlagBadge label="stateful" color="#fbbf24" />}
              {state.is_immutable && <FlagBadge label="immutable" color="#34d399" />}
            </div>
          )}
        </div>

        {/* Bottom: children */}
        <ExpandablePort direction="bottom" label="children"
          items={filteredChildren.map(c => ({ id: c.task_id, name: c.name }))}
          colors={PORT_COLORS.children} expanded={expandedPorts.has('children')}
          onToggle={() => togglePort('children')} onClickItem={(id) => onExpand(id, 'child')}
          existingNodeIds={existingNodeIds} />
      </div>

      {/* Right: cells */}
      <CellPort cells={filteredCells} colors={PORT_COLORS.cells}
        expanded={expandedPorts.has('cells')} onToggle={() => togglePort('cells')}
        expandedCell={expandedCell} cellDetail={cellDetail}
        cellDetailLoading={cellDetailLoading} cellsChanged={data.cellsChanged}
        onClickCell={handleExpandCell} totalCells={data.cells.length} />
    </div>
  )
}

// ─── ExpandablePort ─────────────────────────────────────────

interface ExpandablePortProps {
  direction: 'top' | 'bottom' | 'left' | 'right'
  label: string
  items: { id: number; name: string; label?: string }[]
  colors: { accent: string; glow: string; hoverBg: string }
  expanded: boolean
  onToggle: () => void
  onClickItem: (id: number) => void
  existingNodeIds: Set<number>
}

function ExpandablePort({ direction, label, items, colors, expanded, onToggle, onClickItem, existingNodeIds }: ExpandablePortProps) {
  const [showAll, setShowAll] = useState(false)
  const LIMIT = 7

  if (items.length === 0) return null

  const isVertical = direction === 'top' || direction === 'bottom'
  const isBefore = direction === 'top' || direction === 'left'
  const visibleItems = showAll ? items : items.slice(0, LIMIT)
  const remaining = items.length - LIMIT

  const stub = (
    <button
      className="flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[10px] transition-all hover:scale-105"
      style={{
        background: expanded ? `${colors.accent}15` : 'rgba(0,0,0,0.3)',
        border: `1px solid ${expanded ? colors.accent + '50' : colors.accent + '25'}`,
        color: colors.accent,
      }}
      onClick={(e) => { e.stopPropagation(); onToggle() }}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: colors.accent }} />
      <span>{items.length} {label}</span>
      <span className="text-[8px]">{expanded ? '−' : '+'}</span>
    </button>
  )

  const connector = expanded ? <ConnectorLine direction={isVertical ? 'vertical' : 'horizontal'} color={colors.accent} /> : null

  const list = expanded ? (
    <div className={`flex ${isVertical ? 'flex-col items-center' : 'flex-col'} gap-0.5`}>
      {visibleItems.map((item, i) => {
        const isOnCanvas = existingNodeIds.has(item.id)
        return (
          <button key={`${item.id}-${i}`}
            className="space-port-item flex items-center gap-1.5 rounded bg-zinc-900/60 px-2 py-1"
            style={{ '--port-accent': colors.accent, '--port-glow': colors.glow, '--port-hover-bg': colors.hoverBg, opacity: isOnCanvas ? 0.5 : 1 } as React.CSSProperties}
            onClick={(e) => { e.stopPropagation(); onClickItem(item.id) }}
          >
            <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: isOnCanvas ? '#71717a' : colors.accent }} />
            <span className="max-w-[130px] truncate font-mono text-[10px] text-zinc-300">{middleTruncate(item.name, 22)}</span>
            {item.label && <span className="text-[8px] text-zinc-600">{item.label}</span>}
            {isOnCanvas && <span className="text-[8px] text-zinc-500">&#10003;</span>}
          </button>
        )
      })}
      {remaining > 0 && !showAll && (
        <button className="rounded bg-zinc-900/40 px-2 py-0.5 text-[9px] text-zinc-500 hover:text-zinc-400"
          onClick={(e) => { e.stopPropagation(); setShowAll(true) }}>+{remaining} more</button>
      )}
      {showAll && remaining > 0 && (
        <button className="rounded bg-zinc-900/40 px-2 py-0.5 text-[9px] text-zinc-500 hover:text-zinc-400"
          onClick={(e) => { e.stopPropagation(); setShowAll(false) }}>show less</button>
      )}
    </div>
  ) : null

  const parts = isBefore ? <>{list}{connector}{stub}</> : <>{stub}{connector}{list}</>

  return (
    <div className={`flex ${isVertical ? 'flex-col' : 'flex-row'} items-center gap-0`}>
      {parts}
    </div>
  )
}

// ─── CellPort ───────────────────────────────────────────────

interface CellPortProps {
  cells: CellInfo[]
  colors: { accent: string; glow: string; hoverBg: string }
  expanded: boolean
  onToggle: () => void
  expandedCell: number | null
  cellDetail: CellDetail | null
  cellDetailLoading: boolean
  cellsChanged: Set<number>
  onClickCell: (cellIndex: number) => void
  totalCells: number
}

function CellPort({ cells, colors, expanded, onToggle, expandedCell, cellDetail, cellDetailLoading, cellsChanged, onClickCell, totalCells }: CellPortProps) {
  const [showAll, setShowAll] = useState(false)
  const LIMIT = 7
  if (totalCells === 0) return null
  const visibleCells = showAll ? cells : cells.slice(0, LIMIT)
  const remaining = cells.length - LIMIT

  const stub = (
    <button
      className="flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[10px] transition-all hover:scale-105"
      style={{
        background: expanded ? `${colors.accent}15` : 'rgba(0,0,0,0.3)',
        border: `1px solid ${expanded ? colors.accent + '50' : colors.accent + '25'}`,
        color: colors.accent,
      }}
      onClick={(e) => { e.stopPropagation(); onToggle() }}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: colors.accent }} />
      <span>{totalCells} cells</span>
      <span className="text-[8px]">{expanded ? '−' : '+'}</span>
    </button>
  )

  return (
    <div className="flex flex-row items-center gap-0">
      {stub}
      {expanded && <ConnectorLine direction="horizontal" color={colors.accent} />}
      {expanded && (
        <div className="flex flex-col gap-0.5">
          {visibleCells.map((cell) => (
            <div key={`${cell.type_name}-${cell.cell_index}`}>
              <button
                className={`space-port-item flex w-full items-center gap-1.5 rounded bg-zinc-900/60 px-2 py-1 ${cellsChanged.has(cell.cell_index) ? 'space-port-flash' : ''}`}
                style={{ '--port-accent': colors.accent, '--port-glow': colors.glow, '--port-hover-bg': colors.hoverBg } as React.CSSProperties}
                onClick={(e) => { e.stopPropagation(); onClickCell(cell.cell_index) }}
              >
                <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: cell.has_data ? colors.accent : '#71717a' }} />
                <span className="text-[9px] text-zinc-500">[{cell.cell_index}]</span>
                <span className="max-w-[110px] truncate font-mono text-[9px] text-zinc-300">{middleTruncate(cell.type_name, 18)}</span>
                <span className="ml-auto text-[9px] text-zinc-600">{expandedCell === cell.cell_index ? '\u25BC' : '\u25B6'}</span>
              </button>
              {expandedCell === cell.cell_index && (
                <div className="cell-data-expand ml-2 mt-0.5 mb-1 rounded bg-zinc-900/80 p-2" style={{ maxWidth: 500 }}>
                  {cellDetailLoading ? <div className="text-[9px] text-zinc-500">Loading...</div>
                    : cellDetail ? (
                      <div className="space-y-1">
                        {cellDetail.data_size_bytes != null && <div className="text-[8px] text-zinc-600">{cellDetail.data_size_bytes} bytes</div>}
                        {cellDetail.data_preview
                          ? <CellContentRenderer preview={cellDetail.data_preview} typeName={cell.type_name} />
                          : <div className="text-[9px] text-zinc-500">No preview</div>}
                      </div>
                    ) : <div className="text-[9px] text-zinc-500">No detail</div>}
                </div>
              )}
            </div>
          ))}
          {remaining > 0 && !showAll && (
            <button className="rounded bg-zinc-900/40 px-2 py-0.5 text-[9px] text-zinc-500 hover:text-zinc-400"
              onClick={(e) => { e.stopPropagation(); setShowAll(true) }}>+{remaining} more</button>
          )}
          {showAll && remaining > 0 && (
            <button className="rounded bg-zinc-900/40 px-2 py-0.5 text-[9px] text-zinc-500 hover:text-zinc-400"
              onClick={(e) => { e.stopPropagation(); setShowAll(false) }}>show less</button>
          )}
        </div>
      )}
    </div>
  )
}

// ─── DAG View Components ────────────────────────────────────

function DagNodeLabel({ taskId, isRoot, isSelected, onClick }: {
  taskId: number; isRoot: boolean; isSelected: boolean; onClick: () => void
}) {
  const data = useTaskData(taskId)
  const name = data.state?.name || `Task #${taskId}`
  const displayName = middleTruncate(name, 40)

  return (
    <div
      onClick={(e) => { e.stopPropagation(); onClick() }}
      className="cursor-pointer select-none whitespace-nowrap rounded px-3 py-1.5 font-mono text-[11px] font-medium"
      style={{
        background: isSelected ? '#1d4ed8' : '#27272a',
        color: isSelected ? '#f4f4f5' : '#d4d4d8',
        border: isRoot ? '2px solid #3b82f6' : isSelected ? '1px solid #3b82f6' : '1px solid #3f3f46',
        boxShadow: isSelected ? '0 0 10px rgba(59,130,246,0.3)' : '0 1px 3px rgba(0,0,0,0.4)',
      }}
    >
      {displayName}
    </div>
  )
}

function DagEdgePath({ from, to, type }: {
  from: { x: number; y: number }; to: { x: number; y: number }; type: CanvasEdge['type']
}) {
  const halfW = DAG_NODE_W / 2
  const edgeColor = EDGE_COLORS[type]
  const isForward = to.x - from.x > 10 // target is meaningfully to the right

  let d: string
  if (isForward) {
    // Normal left-to-right edge: exit right side, enter left side
    const x1 = from.x + halfW, y1 = from.y
    const x2 = to.x - halfW, y2 = to.y
    const dx = x2 - x1
    d = `M ${x1} ${y1} C ${x1 + dx * 0.5} ${y1}, ${x2 - dx * 0.5} ${y2}, ${x2} ${y2}`
  } else {
    // Back-edge (cycle): exit left side, arc above/below, enter right side
    const x1 = from.x - halfW, y1 = from.y
    const x2 = to.x + halfW, y2 = to.y
    const arcOffset = 60 + Math.abs(from.x - to.x) * 0.3
    const arcY = Math.min(y1, y2) - arcOffset
    d = `M ${x1} ${y1} C ${x1 - 40} ${arcY}, ${x2 + 40} ${arcY}, ${x2} ${y2}`
  }

  return (
    <g>
      <path d={d} stroke={edgeColor} strokeWidth={1.5} fill="none" opacity={0.45}
        strokeDasharray={isForward ? undefined : '4 3'}
        markerEnd={`url(#dag-arrow-${type})`} />
    </g>
  )
}

// ─── EdgePath ───────────────────────────────────────────────

type EdgeSide = 'top' | 'right' | 'bottom' | 'left'

function getEdgeSides(from: NodePosition, to: NodePosition): { fromSide: EdgeSide; toSide: EdgeSide } {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const angle = Math.atan2(dy, dx) * (180 / Math.PI)

  let fromSide: EdgeSide, toSide: EdgeSide

  // Determine which side of source node to connect from based on angle to target
  if (angle >= -45 && angle < 45) {
    fromSide = 'right'
    toSide = 'left'
  } else if (angle >= 45 && angle < 135) {
    fromSide = 'bottom'
    toSide = 'top'
  } else if (angle >= -135 && angle < -45) {
    fromSide = 'top'
    toSide = 'bottom'
  } else {
    fromSide = 'left'
    toSide = 'right'
  }

  return { fromSide, toSide }
}

function EdgePath({ from, to, type, fromSize, toSize }: {
  from: NodePosition; to: NodePosition; type: CanvasEdge['type']
  fromSize: NodeSize | null; toSize: NodeSize | null
}) {
  // Use fixed card dimensions (ignoring ports) for cleaner edge connections
  // The card itself is ~240-340px wide and ~80-120px tall
  const CARD_HALF_W = 120
  const CARD_HALF_H = 50

  // Calculate dynamic port positions based on node positions
  const { fromSide, toSide } = getEdgeSides(from, to)

  let x1: number, y1: number, x2: number, y2: number

  // Source position based on which side - connect to card edge, not ports
  switch (fromSide) {
    case 'top':
      x1 = from.x; y1 = from.y - CARD_HALF_H
      break
    case 'right':
      x1 = from.x + CARD_HALF_W; y1 = from.y
      break
    case 'bottom':
      x1 = from.x; y1 = from.y + CARD_HALF_H
      break
    case 'left':
      x1 = from.x - CARD_HALF_W; y1 = from.y
      break
  }

  // Target position based on which side - connect to card edge, not ports
  switch (toSide) {
    case 'top':
      x2 = to.x; y2 = to.y - CARD_HALF_H
      break
    case 'right':
      x2 = to.x + CARD_HALF_W; y2 = to.y
      break
    case 'bottom':
      x2 = to.x; y2 = to.y + CARD_HALF_H
      break
    case 'left':
      x2 = to.x - CARD_HALF_W; y2 = to.y
      break
  }

  const dx = x2 - x1
  const dy = y2 - y1
  let cx1: number, cy1: number, cx2: number, cy2: number

  // Control points based on the connection direction
  const isVertical = fromSide === 'top' || fromSide === 'bottom'
  const isHorizontal = fromSide === 'left' || fromSide === 'right'

  if (isVertical) {
    cx1 = x1; cy1 = y1 + dy * 0.4
    cx2 = x2; cy2 = y2 - dy * 0.4
  } else {
    cx1 = x1 + dx * 0.4; cy1 = y1
    cx2 = x2 - dx * 0.4; cy2 = y2
  }

  const d = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`
  const edgeColor = EDGE_COLORS[type]

  return (
    <g>
      <path d={d} stroke={edgeColor} strokeWidth={4} fill="none" opacity={0.08} />
      <path d={d} stroke={edgeColor} strokeWidth={1.5} fill="none" opacity={0.5} strokeDasharray="8 4" markerEnd={`url(#arrow-${type})`} />
    </g>
  )
}

// ─── Helpers ────────────────────────────────────────────────

function FlagBadge({ label, color }: { label: string; color: string }) {
  return (
    <span className="rounded-full px-1.5 py-0.5 text-[8px]"
      style={{ border: `1px solid ${color}40`, color, background: `${color}10` }}>{label}</span>
  )
}

function ConnectorLine({ direction, color }: { direction: 'vertical' | 'horizontal'; color: string }) {
  if (direction === 'vertical') {
    return <div className="flex justify-center"><div className="w-px" style={{ height: 16, background: `linear-gradient(to bottom, ${color}50, ${color}15)` }} /></div>
  }
  return <div className="flex items-center"><div className="h-px" style={{ width: 16, background: `linear-gradient(to right, ${color}15, ${color}50)` }} /></div>
}
