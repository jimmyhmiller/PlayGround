import { useEffect, useRef } from 'react'
import {
  useGraphCanvas,
  GraphCanvasComponent,
  type RenderNodeProps,
} from 'graph-ui'
import 'graph-ui/styles/graph-canvas.css'

let nextId = 1
function genId() {
  return `node-${nextId++}`
}

const NAMES = [
  'Dashboard', 'Settings', 'Profile', 'API', 'Database',
  'Cache', 'Logger', 'Metrics', 'Queue', 'Worker',
  'Scheduler', 'Notifications', 'Search', 'Storage', 'Gateway',
]

function randomName() {
  return NAMES[Math.floor(Math.random() * NAMES.length)]
}

export function App() {
  const canvas = useGraphCanvas<string>({
    edgeTypes: { child: '#60a5fa', dep: '#a855f7' },
    requireModifierKey: false,
  })

  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    // Build initial graph
    const nodes = new Map<string, { x: number; y: number }>()
    const edges: { fromId: string; toId: string; type: string }[] = []

    nodes.set('App', { x: 0, y: 0 })
    nodes.set('Router', { x: -300, y: 400 })
    nodes.set('Auth', { x: 300, y: 400 })
    nodes.set('HomePage', { x: -300, y: 800 })
    nodes.set('LoginPage', { x: 300, y: 800 })

    edges.push({ fromId: 'App', toId: 'Router', type: 'child' })
    edges.push({ fromId: 'App', toId: 'Auth', type: 'child' })
    edges.push({ fromId: 'Router', toId: 'HomePage', type: 'child' })
    edges.push({ fromId: 'Auth', toId: 'LoginPage', type: 'dep' })

    // Skip IDs past existing names
    nextId = 10

    canvas.setNodes(nodes)
    canvas.setEdges(edges)
    canvas.setGraphRoot('App')
    canvas.selectNode('App')
  }, [])

  const handleExpand = (fromId: string, edgeType: 'child' | 'dep') => {
    const id = genId()
    const name = randomName()
    // Store the display name on the id for rendering
    nameMap.current.set(id, name)
    const dir = edgeType === 'child'
      ? { dx: 0, dy: 400 }
      : { dx: 400, dy: 0 }
    canvas.expandNode(fromId, id, dir, edgeType)
  }

  const nameMap = useRef(new Map<string, string>())

  const renderNode = ({ id, isSelected, isRoot }: RenderNodeProps<string>) => {
    const displayName = nameMap.current.get(id) || id
    const classes = [
      'demo-node',
      isSelected && 'selected',
      isRoot && 'root',
    ].filter(Boolean).join(' ')

    return (
      <div className={classes} data-drag-handle={id}>
        <div className="demo-node-name">{displayName}</div>
        <div className="demo-node-buttons">
          <button
            className="demo-node-btn child"
            onClick={() => handleExpand(id, 'child')}
          >
            +child
          </button>
          <button
            className="demo-node-btn dep"
            onClick={() => handleExpand(id, 'dep')}
          >
            +dep
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div className="demo-instructions">
        Drag to pan &middot; Scroll to zoom<br />
        Click node to select &middot; Drag node to move<br />
        <kbd>Delete</kbd> to remove selected node
      </div>
      <GraphCanvasComponent
        canvas={canvas}
        renderNode={renderNode}
        theme={{
          edgeColors: { child: '#60a5fa', dep: '#a855f7' },
        }}
      />
    </div>
  )
}
