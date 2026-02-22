import { useEffect, useRef, useState, useCallback } from 'react'
import type { NodeSize } from '../types'

/**
 * Hook that measures node sizes via ResizeObserver.
 * Returns a ref map of node sizes, a version counter that increments on changes,
 * and a callback to create ref functions for individual nodes.
 */
export function useResizeObserver<Id = string>() {
  const nodeSizesRef = useRef<Map<Id, NodeSize>>(new Map())
  const [sizeVersion, setSizeVersion] = useState(0)
  const observerRef = useRef<ResizeObserver | null>(null)
  const idParserRef = useRef<((s: string) => Id) | null>(null)

  useEffect(() => {
    observerRef.current = new ResizeObserver((entries) => {
      let changed = false
      for (const entry of entries) {
        const el = entry.target as HTMLElement
        const idStr = el.dataset.nodeSize
        if (!idStr) continue
        const nodeId = idParserRef.current
          ? idParserRef.current(idStr)
          : (idStr as unknown as Id)
        const w = Math.round(entry.contentRect.width)
        const h = Math.round(entry.contentRect.height)
        const prev = nodeSizesRef.current.get(nodeId)
        if (!prev || prev.w !== w || prev.h !== h) {
          nodeSizesRef.current.set(nodeId, { w, h })
          changed = true
        }
      }
      if (changed) setSizeVersion((v) => v + 1)
    })
    return () => observerRef.current?.disconnect()
  }, [])

  const observeNodeRef = useCallback(
    (nodeId: Id) => (el: HTMLDivElement | null) => {
      if (el && observerRef.current) {
        el.dataset.nodeSize = String(nodeId)
        observerRef.current.observe(el)
      }
    },
    []
  )

  const setIdParser = useCallback((parser: (s: string) => Id) => {
    idParserRef.current = parser
  }, [])

  return { nodeSizesRef, sizeVersion, observeNodeRef, setIdParser }
}
