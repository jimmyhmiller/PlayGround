import { useCallback, useRef, useState } from 'react'
import { controlApi, DebugEvent } from '@/lib/control-api'
import { usePoll } from './use-poll'

const POLL_MS = 250
const MAX_EVENTS = 500

export function useEventLog() {
  const [events, setEvents] = useState<DebugEvent[]>([])
  const sinceRef = useRef(0)

  const fetcher = useCallback(async () => {
    return controlApi.events(sinceRef.current)
  }, [])

  const onData = useCallback((newEvents: DebugEvent[]) => {
    if (newEvents.length === 0) return
    sinceRef.current = newEvents[newEvents.length - 1].seq
    setEvents(prev => {
      const combined = [...prev, ...newEvents]
      return combined.length > MAX_EVENTS ? combined.slice(-MAX_EVENTS) : combined
    })
  }, [])

  usePoll(fetcher, onData, POLL_MS)

  const clearEvents = useCallback(() => {
    setEvents([])
  }, [])

  return { events, clearEvents }
}
