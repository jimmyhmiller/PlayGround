import { useCallback, useState } from 'react'
import { controlApi, ControlStatus, PendingTaskInfo } from '@/lib/control-api'
import { usePoll } from './use-poll'

const POLL_MS = 200

export function useControlStatus() {
  const [status, setStatus] = useState<ControlStatus | null>(null)
  const [pending, setPending] = useState<PendingTaskInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  const fetcher = useCallback(async () => {
    const [s, p] = await Promise.all([controlApi.status(), controlApi.pending()])
    return { status: s, pending: p }
  }, [])

  const onData = useCallback((data: { status: ControlStatus; pending: PendingTaskInfo[] }) => {
    setStatus(data.status)
    setPending(data.pending)
    setError(null)
  }, [])

  const poll = usePoll(fetcher, onData, POLL_MS)

  const pause = useCallback(async () => {
    await controlApi.pause()
    poll()
  }, [poll])

  const resume = useCallback(async () => {
    await controlApi.resume()
    poll()
  }, [poll])

  const step = useCallback(async (count = 1) => {
    await controlApi.step(count)
    poll()
  }, [poll])

  const stepTask = useCallback(async (taskId: number) => {
    await controlApi.stepTask(taskId)
    poll()
  }, [poll])

  const addBreakpoint = useCallback(async (pattern: string) => {
    await controlApi.addBreakpoint(pattern)
    poll()
  }, [poll])

  const removeBreakpoint = useCallback(async (id: number) => {
    await controlApi.removeBreakpoint(id)
    poll()
  }, [poll])

  const toggleBreakpoint = useCallback(async (id: number, enabled: boolean) => {
    await controlApi.toggleBreakpoint(id, enabled)
    poll()
  }, [poll])

  return {
    status,
    pending,
    error,
    pause,
    resume,
    step,
    stepTask,
    addBreakpoint,
    removeBreakpoint,
    toggleBreakpoint,
    poll,
  }
}
