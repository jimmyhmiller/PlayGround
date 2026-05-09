import { useCallback, useEffect, useRef, useState } from 'react'
import {
  controlApi,
  TaskStateInfo,
  CellInfo,
  ChildInfo,
  TaskDepsInfo,
} from '@/lib/control-api'

export interface TaskData {
  state: TaskStateInfo | null
  cells: CellInfo[]
  children: ChildInfo[]
  deps: TaskDepsInfo | null
  loading: boolean
  stateChanged: boolean
  cellsChanged: Set<number>
}

export function useTaskData(taskId: number | null): TaskData {
  const [state, setState] = useState<TaskStateInfo | null>(null)
  const [cells, setCells] = useState<CellInfo[]>([])
  const [children, setChildren] = useState<ChildInfo[]>([])
  const [deps, setDeps] = useState<TaskDepsInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [stateChanged, setStateChanged] = useState(false)
  const [cellsChanged, setCellsChanged] = useState<Set<number>>(new Set())

  const prevState = useRef<TaskStateInfo | null>(null)
  const prevCells = useRef<CellInfo[]>([])
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = useCallback(async (id: number) => {
    const [stateRes, cellsRes, childrenRes, depsRes] = await Promise.allSettled([
      controlApi.taskState(id),
      controlApi.liveCells(id),
      controlApi.liveChildren(id),
      controlApi.liveDeps(id),
    ])

    const newState = stateRes.status === 'fulfilled' ? stateRes.value : null
    const newCells = cellsRes.status === 'fulfilled' ? cellsRes.value : []
    const newChildren = childrenRes.status === 'fulfilled' ? childrenRes.value : []
    const newDeps = depsRes.status === 'fulfilled' ? depsRes.value : null

    // Detect state change
    if (prevState.current && newState) {
      if (prevState.current.state !== newState.state ||
          prevState.current.is_dirty !== newState.is_dirty ||
          prevState.current.is_in_progress !== newState.is_in_progress) {
        setStateChanged(true)
        setTimeout(() => setStateChanged(false), 600)
      }
    }

    // Detect cell changes
    if (prevCells.current.length > 0 && newCells.length > 0) {
      const changed = new Set<number>()
      for (const cell of newCells) {
        const prev = prevCells.current.find(c => c.cell_index === cell.cell_index)
        if (prev && prev.has_data !== cell.has_data) {
          changed.add(cell.cell_index)
        }
      }
      // New cells that didn't exist before
      if (newCells.length !== prevCells.current.length) {
        for (const cell of newCells) {
          if (!prevCells.current.find(c => c.cell_index === cell.cell_index)) {
            changed.add(cell.cell_index)
          }
        }
      }
      if (changed.size > 0) {
        setCellsChanged(changed)
        setTimeout(() => setCellsChanged(new Set()), 600)
      }
    }

    prevState.current = newState
    prevCells.current = newCells

    setState(newState)
    setCells(newCells)
    setChildren(newChildren)
    setDeps(newDeps)
    setLoading(false)
  }, [])

  useEffect(() => {
    if (taskId == null) {
      setState(null)
      setCells([])
      setChildren([])
      setDeps(null)
      setLoading(false)
      prevState.current = null
      prevCells.current = []
      return
    }

    // Reset on task change
    setLoading(true)
    setStateChanged(false)
    setCellsChanged(new Set())
    prevState.current = null
    prevCells.current = []

    fetchAll(taskId)

    intervalRef.current = setInterval(() => fetchAll(taskId), 500)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [taskId, fetchAll])

  return { state, cells, children, deps, loading, stateChanged, cellsChanged }
}
