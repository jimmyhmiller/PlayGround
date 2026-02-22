import type { EdgePathProps } from '../types'
import { getEdgeSides } from '../layout/edge-sides'

const CARD_HALF_W = 120
const CARD_HALF_H = 50

export function EdgePath({
  from,
  to,
  type,
  fromSize,
  toSize,
  color,
  edgeWidth = 1.5,
  glowWidth = 4,
  edgeOpacity = 0.5,
  glowOpacity = 0.08,
  dashArray = '8 4',
  arrowId,
}: EdgePathProps) {
  const { fromSide, toSide } = getEdgeSides(from, to)

  // Use measured sizes or fallback to card dimensions
  const fromHalfW = fromSize ? fromSize.w / 2 : CARD_HALF_W
  const fromHalfH = fromSize ? fromSize.h / 2 : CARD_HALF_H
  const toHalfW = toSize ? toSize.w / 2 : CARD_HALF_W
  const toHalfH = toSize ? toSize.h / 2 : CARD_HALF_H

  // Clamp to card-like proportions to keep edges connected to the card body, not ports
  const fHW = Math.min(fromHalfW, CARD_HALF_W)
  const fHH = Math.min(fromHalfH, CARD_HALF_H)
  const tHW = Math.min(toHalfW, CARD_HALF_W)
  const tHH = Math.min(toHalfH, CARD_HALF_H)

  let x1: number, y1: number
  switch (fromSide) {
    case 'top':    x1 = from.x; y1 = from.y - fHH; break
    case 'right':  x1 = from.x + fHW; y1 = from.y; break
    case 'bottom': x1 = from.x; y1 = from.y + fHH; break
    case 'left':   x1 = from.x - fHW; y1 = from.y; break
  }

  let x2: number, y2: number
  switch (toSide) {
    case 'top':    x2 = to.x; y2 = to.y - tHH; break
    case 'right':  x2 = to.x + tHW; y2 = to.y; break
    case 'bottom': x2 = to.x; y2 = to.y + tHH; break
    case 'left':   x2 = to.x - tHW; y2 = to.y; break
  }

  const dx = x2 - x1
  const dy = y2 - y1
  const isVertical = fromSide === 'top' || fromSide === 'bottom'

  let cx1: number, cy1: number, cx2: number, cy2: number
  if (isVertical) {
    cx1 = x1; cy1 = y1 + dy * 0.4
    cx2 = x2; cy2 = y2 - dy * 0.4
  } else {
    cx1 = x1 + dx * 0.4; cy1 = y1
    cx2 = x2 - dx * 0.4; cy2 = y2
  }

  const d = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`

  return (
    <g>
      <path d={d} stroke={color} strokeWidth={glowWidth} fill="none" opacity={glowOpacity} />
      <path
        d={d}
        stroke={color}
        strokeWidth={edgeWidth}
        fill="none"
        opacity={edgeOpacity}
        strokeDasharray={dashArray}
        markerEnd={`url(#${arrowId})`}
      />
    </g>
  )
}
