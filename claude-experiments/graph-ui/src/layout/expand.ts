import type { NodePosition } from '../types'
import { isOccupied } from './collision'

/**
 * Compute the position for a new node expanded from an existing node.
 *
 * @param fromPos - Position of the source node
 * @param direction - Direction vector for expansion (e.g. {dx: 0, dy: 600} for downward)
 * @param existingPositions - Iterator over all existing node positions
 * @param spacing - Base spacing between nodes
 * @param collisionThreshold - Minimum distance to avoid overlap
 * @returns The computed position for the new node
 */
export function computeExpandPosition(
  fromPos: NodePosition,
  direction: { dx: number; dy: number },
  existingPositions: Iterable<NodePosition>,
  spacing: number,
  collisionThreshold: number
): NodePosition {
  let x = fromPos.x + direction.dx
  let y = fromPos.y + direction.dy

  // Determine slide axis: if primary direction is vertical, slide horizontally and vice versa
  const isVertical = Math.abs(direction.dy) > Math.abs(direction.dx)

  let attempts = 0
  while (isOccupied(x, y, existingPositions, collisionThreshold) && attempts < 20) {
    if (isVertical) {
      x += spacing * 0.6
    } else {
      y += spacing * 0.6
    }
    attempts++
  }

  return { x, y }
}

/**
 * Find a free position near a target point, sliding along the given axis.
 */
export function findFreePosition(
  startX: number,
  startY: number,
  slideAxis: 'x' | 'y',
  existingPositions: Iterable<NodePosition>,
  spacing: number,
  collisionThreshold: number
): NodePosition {
  let x = startX
  let y = startY
  let attempts = 0

  while (isOccupied(x, y, existingPositions, collisionThreshold) && attempts < 20) {
    if (slideAxis === 'x') {
      x += spacing * 0.6
    } else {
      y += spacing * 0.6
    }
    attempts++
  }

  return { x, y }
}
