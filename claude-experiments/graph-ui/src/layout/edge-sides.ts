import type { NodePosition, EdgeSide } from '../types'

/**
 * Determine which side of each node an edge should connect to,
 * based on the angle between the two node positions.
 */
export function getEdgeSides(
  from: NodePosition,
  to: NodePosition
): { fromSide: EdgeSide; toSide: EdgeSide } {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const angle = Math.atan2(dy, dx) * (180 / Math.PI)

  let fromSide: EdgeSide, toSide: EdgeSide

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
