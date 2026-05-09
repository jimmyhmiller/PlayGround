import { useCallback, useEffect, useRef } from 'react'

export function usePoll<T>(
  fetcher: () => Promise<T>,
  onData: (data: T) => void,
  intervalMs: number,
  deps: unknown[] = [],
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const poll = useCallback(async () => {
    try {
      const data = await fetcher()
      onData(data)
    } catch {
      // silently ignore polling errors
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    poll()
    intervalRef.current = setInterval(poll, intervalMs)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [poll, intervalMs])

  return poll
}
