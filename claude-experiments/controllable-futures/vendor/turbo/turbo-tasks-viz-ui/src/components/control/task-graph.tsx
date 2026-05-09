'use client'

import { useCallback, useEffect, useState } from 'react'
import {
  ReactFlow,
  Node,
  Edge,
  Position,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MarkerType,
  Handle,
  NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import dagre from '@dagrejs/dagre'
import { controlApi, TaskGraph } from '@/lib/control-api'
import { Button } from '@/components/ui/button'

interface Props {
  taskId: number
  onSelectTask: (taskId: number) => void
}

interface TaskNodeData {
  label: string
  state: string
  taskId: number
  style: {
    background: string
    color: string
    border: string
    borderRadius: string
    padding: string
    fontSize: string
    width: number
    overflow: 'hidden'
    textOverflow: 'ellipsis'
    whiteSpace: 'nowrap'
  }
}

const NODE_WIDTH = 220
const NODE_HEIGHT = 40

const STATE_COLORS: Record<string, string> = {
  created: '#52525b',     // zinc-600
  scheduled: '#ca8a04',   // yellow-600
  in_progress: '#2563eb', // blue-600
  completed: '#16a34a',   // green-600
  dirty: '#dc2626',       // red-600
}

const EDGE_STYLES: Record<string, { stroke: string; strokeDasharray?: string }> = {
  child: { stroke: '#3b82f6' },
  output_dep: { stroke: '#22c55e', strokeDasharray: '5 3' },
  cell_dep: { stroke: '#a855f7', strokeDasharray: '2 2' },
}

// Custom node component with handles on all sides
function TaskNode({ data, selected }: NodeProps) {
  const nodeData = data as unknown as TaskNodeData
  return (
    <div
      style={{
        background: nodeData.style.background,
        color: nodeData.style.color,
        border: nodeData.style.border,
        borderRadius: nodeData.style.borderRadius,
        padding: nodeData.style.padding,
        fontSize: nodeData.style.fontSize,
        width: nodeData.style.width,
        overflow: nodeData.style.overflow,
        textOverflow: nodeData.style.textOverflow,
        whiteSpace: nodeData.style.whiteSpace,
        boxShadow: selected ? '0 0 0 2px #fff' : undefined,
      }}
    >
      {/* Handles on all four sides */}
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Right} style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Bottom} style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />

      <Handle type="source" position={Position.Top} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Left} style={{ opacity: 0 }} />

      {nodeData.label}
    </div>
  )
}

const nodeTypes = {
  task: TaskNode,
}

function getEdgePosition(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number
): { sourcePosition: Position; targetPosition: Position } {
  const dx = targetX - sourceX
  const dy = targetY - sourceY

  // Determine source position (where edge leaves source node)
  let sourcePosition: Position
  let targetPosition: Position

  // Use angle to determine best positions
  const angle = Math.atan2(dy, dx) * (180 / Math.PI)

  // Source position (where the edge starts)
  if (angle >= -45 && angle < 45) {
    sourcePosition = Position.Right
  } else if (angle >= 45 && angle < 135) {
    sourcePosition = Position.Bottom
  } else if (angle >= -135 && angle < -45) {
    sourcePosition = Position.Top
  } else {
    sourcePosition = Position.Left
  }

  // Target position (where the edge ends) - opposite of source
  if (angle >= -45 && angle < 45) {
    targetPosition = Position.Left
  } else if (angle >= 45 && angle < 135) {
    targetPosition = Position.Top
  } else if (angle >= -135 && angle < -45) {
    targetPosition = Position.Bottom
  } else {
    targetPosition = Position.Right
  }

  return { sourcePosition, targetPosition }
}

function layoutGraph(graph: TaskGraph): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 30, ranksep: 50 })

  for (const node of graph.nodes) {
    g.setNode(String(node.task_id), { width: NODE_WIDTH, height: NODE_HEIGHT })
  }

  for (const edge of graph.edges) {
    g.setEdge(String(edge.source), String(edge.target))
  }

  dagre.layout(g)

  // Create position map for edge calculation
  const nodePositions = new Map<string, { x: number; y: number }>()
  graph.nodes.forEach((n) => {
    const pos = g.node(String(n.task_id))
    nodePositions.set(String(n.task_id), { x: pos.x, y: pos.y })
  })

  const nodes: Node[] = graph.nodes.map((n) => {
    const pos = g.node(String(n.task_id))
    return {
      id: String(n.task_id),
      type: 'task',
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
      data: {
        label: `#${n.task_id} ${n.name}`,
        state: n.state,
        taskId: n.task_id,
        style: {
          background: STATE_COLORS[n.state] || STATE_COLORS.created,
          color: '#fff',
          border: n.task_id === graph.root ? '2px solid #fff' : '1px solid #555',
          borderRadius: '6px',
          padding: '6px 10px',
          fontSize: '11px',
          width: NODE_WIDTH,
          overflow: 'hidden' as const,
          textOverflow: 'ellipsis' as const,
          whiteSpace: 'nowrap' as const,
        },
      },
    }
  })

  const edges: Edge[] = graph.edges.map((e, i) => {
    const style = EDGE_STYLES[e.edge_type] || EDGE_STYLES.child

    // Calculate dynamic positions
    const sourcePos = nodePositions.get(String(e.source))!
    const targetPos = nodePositions.get(String(e.target))!
    const positions = getEdgePosition(sourcePos.x, sourcePos.y, targetPos.x, targetPos.y)

    return {
      id: `e-${e.source}-${e.target}-${i}`,
      source: String(e.source),
      target: String(e.target),
      sourcePosition: positions.sourcePosition,
      targetPosition: positions.targetPosition,
      type: 'smoothstep',
      style,
      label: e.label || undefined,
      labelStyle: { fontSize: '9px', fill: '#888' },
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: style.stroke },
      animated: e.edge_type === 'cell_dep',
    }
  })

  return { nodes, edges }
}

export function TaskGraphView({ taskId, onSelectTask }: Props) {
  const [graph, setGraph] = useState<TaskGraph | null>(null)
  const [depth, setDepth] = useState(2)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])

  useEffect(() => {
    setLoading(true)
    setError(null)
    controlApi.taskGraph(taskId, depth).then((g) => {
      setGraph(g)
      const layout = layoutGraph(g)
      setNodes(layout.nodes)
      setEdges(layout.edges)
      setLoading(false)
    }).catch((e) => {
      setError(String(e))
      setLoading(false)
    })
  }, [taskId, depth])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const id = Number(node.id)
    if (!isNaN(id)) onSelectTask(id)
  }, [onSelectTask])

  if (loading) return <div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading graph...</div>
  if (error) return <div className="flex h-full items-center justify-center text-xs text-red-400">{error}</div>
  if (!graph || graph.nodes.length === 0) return <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No graph data</div>

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b px-3 py-1.5">
        <span className="text-[10px] text-muted-foreground">Depth:</span>
        {[1, 2, 3, 4].map((d) => (
          <Button
            key={d}
            variant={depth === d ? 'default' : 'outline'}
            size="sm"
            className="h-5 px-2 text-[10px]"
            onClick={() => setDepth(d)}
          >
            {d}
          </Button>
        ))}
        <span className="ml-auto text-[10px] text-muted-foreground">
          {graph.nodes.length} nodes, {graph.edges.length} edges
        </span>
      </div>
      <div className="flex items-center gap-3 border-b px-3 py-1">
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-4 rounded" style={{ background: '#3b82f6' }} /> child
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-4 rounded border border-dashed" style={{ borderColor: '#22c55e' }} /> output dep
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-4 rounded border border-dotted" style={{ borderColor: '#a855f7' }} /> cell dep
        </span>
      </div>
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          proOptions={{ hideAttribution: true }}
          className="bg-zinc-950"
        >
          <Controls className="[&>button]:bg-zinc-800 [&>button]:text-zinc-300 [&>button]:border-zinc-700" />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#333" />
        </ReactFlow>
      </div>
    </div>
  )
}
