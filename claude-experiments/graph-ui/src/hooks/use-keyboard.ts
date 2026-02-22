import { useEffect } from 'react'

/**
 * Hook that handles Delete/Backspace key to remove the selected node.
 * Ignores key events when focused on input/textarea elements.
 */
export function useKeyboard<Id>(
  selectedNodeId: Id | null,
  graphRoot: Id | null,
  removeNode: (id: Id) => void
) {
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
}
