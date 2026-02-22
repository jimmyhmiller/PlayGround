import type { NodePosition } from '../types'

/**
 * Check if a position overlaps with any existing node position
 * within the given threshold distance.
 */
export function isOccupied(
  x: number,
  y: number,
  positions: Iterable<NodePosition>,
  threshold: number
): boolean {
  for (const pos of positions) {
    if (Math.abs(pos.x - x) < threshold && Math.abs(pos.y - y) < threshold) {
      return true
    }
  }
  return false
}
