'use client'

import { useState } from 'react'
import { controlApi, ActiveTask } from '@/lib/control-api'
import { usePoll } from './use-poll'

const POLL_MS = 300

export function useActiveTasks() {
  const [tasks, setTasks] = useState<ActiveTask[]>([])

  usePoll(
    () => controlApi.activeTasks(),
    (data) => setTasks(data),
    POLL_MS,
  )

  return { tasks }
}
