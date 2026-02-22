import type { GraphCanvasProps } from '../types'
import { mergeTheme } from '../theme/merge-theme'
import { GraphThemeProvider } from '../theme/theme-context'
import { EdgeLayer } from './edge-layer'
import { DefaultControls } from './default-controls'

const DEFAULT_MIN_ZOOM = 0.15
const DEFAULT_MAX_ZOOM = 3

export function GraphCanvas<Id = string>({
  canvas,
  renderNode,
  theme: themeOverride,
  showDefaultControls = true,
  renderControls,
  renderEmpty,
  className,
  style,
}: GraphCanvasProps<Id>) {
  const theme = mergeTheme(themeOverride)

  const {
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
    handlers,
    observeNodeRef,
    resetView,
    clearGraph,
  } = canvas

  const nodeList = Array.from(nodes.entries())

  if (nodes.size === 0 && renderEmpty) {
    return (
      <GraphThemeProvider theme={theme}>
        <div
          className={`graph-canvas-bg ${className || ''}`}
          style={{
            position: 'relative',
            width: '100%',
            height: '100%',
            overflow: 'hidden',
            background: theme.background,
            ...style,
          }}
        >
          {renderEmpty()}
        </div>
      </GraphThemeProvider>
    )
  }

  const transitionStyle = isAnimatingPan
    ? `transform ${theme.panTransitionMs}ms ${theme.panTransitionEasing}`
    : 'none'

  return (
    <GraphThemeProvider theme={theme}>
      <div
        className={`graph-canvas-bg ${className || ''}`}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          background: theme.background,
          ...style,
        }}
      >
        {/* Controls */}
        {renderControls
          ? renderControls()
          : showDefaultControls && (
              <DefaultControls
                zoom={zoom}
                nodeCount={nodes.size}
                minZoom={DEFAULT_MIN_ZOOM}
                maxZoom={DEFAULT_MAX_ZOOM}
                onZoomIn={() => {
                  const next = Math.min(DEFAULT_MAX_ZOOM, canvas.zoom * 1.2)
                  canvas.setZoom(next)
                }}
                onZoomOut={() => {
                  const next = Math.max(DEFAULT_MIN_ZOOM, canvas.zoom / 1.2)
                  canvas.setZoom(next)
                }}
                onResetView={resetView}
                onClearGraph={clearGraph}
                theme={themeOverride}
              />
            )}

        {/* Canvas interaction area */}
        <div
          ref={containerRef as React.RefObject<HTMLDivElement>}
          style={{
            width: '100%',
            height: '100%',
            cursor: isDragging ? theme.cursorDragging : theme.cursorDefault,
          }}
          onMouseDown={handlers.onMouseDown}
          onMouseMove={handlers.onMouseMove}
          onMouseUp={handlers.onMouseUp}
          onMouseLeave={handlers.onMouseUp}
          onWheel={handlers.onWheel}
        >
          {/* World transform container */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: `translate(calc(-50% + ${pan.x}px), calc(-50% + ${pan.y}px)) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: transitionStyle,
            }}
          >
            {/* Edge layer */}
            <EdgeLayer
              edges={edges}
              nodes={nodes}
              nodeSizes={nodeSizesRef.current ?? new Map()}
              edgeColors={theme.edgeColors}
              arrowSize={theme.arrowSize}
              arrowOpacity={theme.arrowOpacity}
              edgeWidth={theme.edgeWidth}
              edgeGlowWidth={theme.edgeGlowWidth}
              edgeOpacity={theme.edgeOpacity}
              edgeGlowOpacity={theme.edgeGlowOpacity}
              edgeDashArray={theme.edgeDashArray}
              sizeVersion={sizeVersion}
            />

            {/* Nodes */}
            {nodeList.map(([nodeId, pos]) => (
              <div
                key={String(nodeId)}
                ref={observeNodeRef(nodeId)}
                style={{
                  position: 'absolute',
                  left: pos.x,
                  top: pos.y,
                  transform: 'translate(-50%, -50%)',
                }}
              >
                {renderNode({
                  id: nodeId,
                  isSelected: nodeId === selectedNodeId,
                  isRoot: nodeId === graphRoot,
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </GraphThemeProvider>
  )
}
