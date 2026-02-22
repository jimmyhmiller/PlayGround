import type { EdgeLayerProps } from '../types'
import { EdgePath } from './edge-path'

export function EdgeLayer<Id = string>({
  edges,
  nodes,
  nodeSizes,
  edgeColors,
  arrowSize = { width: 8, height: 5 },
  arrowOpacity = 0.7,
  edgeWidth,
  edgeGlowWidth,
  edgeOpacity,
  edgeGlowOpacity,
  edgeDashArray,
}: EdgeLayerProps<Id>) {
  // Collect all unique edge types from both the config and actual edges
  const allTypes = new Set<string>(Object.keys(edgeColors))
  for (const edge of edges) {
    allTypes.add(edge.type)
  }

  return (
    <svg
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        width: 0,
        height: 0,
        overflow: 'visible',
        pointerEvents: 'none',
      }}
    >
      <defs>
        {Array.from(allTypes).map((type) => {
          const color = edgeColors[type] || '#888'
          return (
            <marker
              key={type}
              id={`graph-ui-arrow-${type}`}
              viewBox="0 0 10 6"
              refX="9"
              refY="3"
              markerWidth={arrowSize.width}
              markerHeight={arrowSize.height}
              orient="auto"
            >
              <path d="M 0 0 L 10 3 L 0 6 Z" fill={color} opacity={arrowOpacity} />
            </marker>
          )
        })}
      </defs>
      {edges.map((edge, i) => {
        const from = nodes.get(edge.fromId)
        const to = nodes.get(edge.toId)
        if (!from || !to) return null
        const fromSize = nodeSizes.get(edge.fromId) || null
        const toSize = nodeSizes.get(edge.toId) || null
        const color = edgeColors[edge.type] || '#888'
        return (
          <EdgePath
            key={`${String(edge.fromId)}-${String(edge.toId)}-${i}`}
            from={from}
            to={to}
            type={edge.type}
            fromSize={fromSize}
            toSize={toSize}
            color={color}
            edgeWidth={edgeWidth}
            glowWidth={edgeGlowWidth}
            edgeOpacity={edgeOpacity}
            glowOpacity={edgeGlowOpacity}
            dashArray={edgeDashArray}
            arrowId={`graph-ui-arrow-${edge.type}`}
          />
        )
      })}
    </svg>
  )
}
