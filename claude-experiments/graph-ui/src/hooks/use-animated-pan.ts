import { useState, useRef, useCallback } from 'react'

/**
 * Hook that manages animated pan transitions.
 * Returns a flag for whether animation is active, and a function to trigger
 * an animated pan that auto-disables after the transition completes.
 */
export function useAnimatedPan(
  setPan: React.Dispatch<React.SetStateAction<{ x: number; y: number }>>,
  transitionMs: number = 400
) {
  const [isAnimatingPan, setIsAnimatingPan] = useState(false)
  const animatingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const animatePanTo = useCallback(
    (target: { x: number; y: number }) => {
      if (animatingTimerRef.current) clearTimeout(animatingTimerRef.current)
      setIsAnimatingPan(true)
      setPan(target)
      animatingTimerRef.current = setTimeout(
        () => setIsAnimatingPan(false),
        transitionMs + 20
      )
    },
    [setPan, transitionMs]
  )

  const cancelAnimation = useCallback(() => {
    setIsAnimatingPan(false)
    if (animatingTimerRef.current) {
      clearTimeout(animatingTimerRef.current)
      animatingTimerRef.current = null
    }
  }, [])

  return { isAnimatingPan, animatePanTo, cancelAnimation }
}
