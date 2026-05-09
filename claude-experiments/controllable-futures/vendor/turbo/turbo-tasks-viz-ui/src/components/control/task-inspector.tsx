'use client'

import { useEffect, useState } from 'react'
import { controlApi, CellInfo, ChildInfo, TaskDepsInfo, TaskStateInfo, CellDetail } from '@/lib/control-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CellContentRenderer } from './cell-renderer'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'

interface Props {
  taskId: number
  onSelectTask: (taskId: number) => void
}

const STATE_BADGE_COLORS: Record<string, string> = {
  created: 'bg-zinc-600',
  scheduled: 'bg-yellow-600',
  in_progress: 'bg-blue-600',
  completed: 'bg-green-600',
  dirty: 'bg-red-600',
}

export function TaskInspector({ taskId, onSelectTask }: Props) {
  const [stateInfo, setStateInfo] = useState<TaskStateInfo | null>(null)
  const [cells, setCells] = useState<CellInfo[]>([])
  const [children, setChildren] = useState<ChildInfo[]>([])
  const [deps, setDeps] = useState<TaskDepsInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedCell, setExpandedCell] = useState<number | null>(null)
  const [cellDetail, setCellDetail] = useState<CellDetail | null>(null)
  const [cellDetailLoading, setCellDetailLoading] = useState(false)
  const [history, setHistory] = useState<number[]>([])

  useEffect(() => {
    setLoading(true)
    setError(null)
    setExpandedCell(null)
    setCellDetail(null)

    const fetchState = controlApi.taskState(taskId).then(s => {
      setStateInfo(s)
    }).catch(e => console.error('taskState failed:', e))

    const fetchCells = controlApi.liveCells(taskId).then(c => {
      setCells(c)
    }).catch(e => console.error('liveCells failed:', e))

    const fetchChildren = controlApi.liveChildren(taskId).then(ch => {
      setChildren(ch)
    }).catch(e => console.error('liveChildren failed:', e))

    const fetchDeps = controlApi.liveDeps(taskId).then(d => {
      setDeps(d)
    }).catch(e => console.error('liveDeps failed:', e))

    Promise.allSettled([fetchState, fetchCells, fetchChildren, fetchDeps])
      .then(results => {
        const failures = results.filter(r => r.status === 'rejected')
        if (failures.length > 0) {
          setError(`${failures.length} of 4 API calls failed`)
        }
        setLoading(false)
      })
  }, [taskId])

  const handleSelectTask = (id: number) => {
    setHistory(prev => [...prev, taskId])
    onSelectTask(id)
  }

  const handleBack = () => {
    if (history.length > 0) {
      const prev = history[history.length - 1]
      setHistory(h => h.slice(0, -1))
      onSelectTask(prev)
    }
  }

  const handleExpandCell = (cellIndex: number) => {
    if (expandedCell === cellIndex) {
      setExpandedCell(null)
      setCellDetail(null)
      return
    }
    setExpandedCell(cellIndex)
    setCellDetailLoading(true)
    controlApi.cellDetail(taskId, cellIndex).then(d => {
      setCellDetail(d)
      setCellDetailLoading(false)
    }).catch(e => {
      console.error('cellDetail failed:', e)
      setCellDetailLoading(false)
    })
  }

  if (loading) return <div className="p-4 text-xs text-muted-foreground">Loading...</div>

  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-3">
        {/* Header with back button */}
        <div className="flex items-center gap-2">
          {history.length > 0 && (
            <Button variant="ghost" size="sm" className="h-5 px-1 text-[10px]" onClick={handleBack}>
              &larr; Back
            </Button>
          )}
          <h3 className="text-sm font-semibold">Task #{taskId}</h3>
        </div>
        {stateInfo?.name && <div className="break-all text-xs text-muted-foreground">{stateInfo.name}</div>}
        {error && <div className="text-xs text-red-400">{error}</div>}

        {/* State section */}
        {stateInfo && (
          <Section title="State">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge className={`${STATE_BADGE_COLORS[stateInfo.state] || 'bg-zinc-600'} text-[10px]`}>
                {stateInfo.state}
              </Badge>
              {stateInfo.is_dirty && <Badge variant="outline" className="text-[10px] border-red-500 text-red-400">dirty</Badge>}
              {stateInfo.is_in_progress && <Badge variant="outline" className="text-[10px] border-blue-500 text-blue-400">in_progress</Badge>}
              {stateInfo.is_stateful && <Badge variant="outline" className="text-[10px] border-amber-500 text-amber-400">stateful</Badge>}
              {stateInfo.is_immutable && <Badge variant="outline" className="text-[10px] border-emerald-500 text-emerald-400">immutable</Badge>}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] text-muted-foreground">
              <div>cells: {stateInfo.cell_count}</div>
              <div>children: {stateInfo.child_count}</div>
              <div>→ reads output of: {stateInfo.output_dep_count}</div>
              <div>→ reads cells from: {stateInfo.cell_dep_count}</div>
              <div>← read by: {stateInfo.dependent_count}</div>
              <div>output: {stateInfo.has_output ? 'yes' : 'no'}</div>
            </div>
          </Section>
        )}

        {/* Output section */}
        {stateInfo?.has_output && stateInfo.output_description && (
          <Section title="Output">
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-zinc-900 p-2 font-mono text-[10px] text-zinc-300">
              {stateInfo.output_description}
            </pre>
          </Section>
        )}

        {/* Cells section with expandable rows */}
        <Section title={`Cells (${cells.length})`}>
          {cells.length === 0 ? (
            <Empty>No cells</Empty>
          ) : (
            <div className="space-y-0.5">
              {cells.map((c, i) => (
                <div key={i}>
                  <div
                    className="flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-xs hover:bg-zinc-800"
                    onClick={() => handleExpandCell(c.cell_index)}
                  >
                    <span className="text-[10px] text-muted-foreground">[{c.cell_index}]</span>
                    <code className="text-[10px]">{c.type_name}</code>
                    <span className="ml-auto text-[10px] text-muted-foreground">{expandedCell === c.cell_index ? '\u25BC' : '\u25B6'}</span>
                  </div>
                  {expandedCell === c.cell_index && (
                    <div className="ml-4 mt-1 mb-2">
                      {cellDetailLoading ? (
                        <div className="text-[10px] text-muted-foreground">Loading cell data...</div>
                      ) : cellDetail ? (
                        <div className="space-y-1">
                          {cellDetail.data_size_bytes !== null && (
                            <div className="text-[10px] text-muted-foreground">
                              Size: {cellDetail.data_size_bytes} bytes (debug repr)
                            </div>
                          )}
                          {cellDetail.data_preview ? (
                            <CellContentRenderer preview={cellDetail.data_preview} typeName={c.type_name} />
                          ) : (
                            <div className="text-[10px] text-muted-foreground">No data preview available</div>
                          )}
                        </div>
                      ) : (
                        <div className="text-[10px] text-muted-foreground">No detail available</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* Children section */}
        <Section title={`Children (${children.length})`}>
          {children.length === 0 ? (
            <Empty>No children</Empty>
          ) : (
            <div className="space-y-0.5">
              {children.map(c => (
                <TaskLink key={c.task_id} taskId={c.task_id} name={c.name} onClick={handleSelectTask} />
              ))}
            </div>
          )}
        </Section>

        {/* Dependency sections */}
        {deps && (
          <>
            <Section title={`→ Reads Output Of (${deps.output_deps.length})`}>
              {deps.output_deps.length === 0 ? (
                <Empty>None</Empty>
              ) : (
                <div className="space-y-0.5">
                  {deps.output_deps.map(([id, name], i) => (
                    <TaskLink key={`${id}-${i}`} taskId={id} name={name} onClick={handleSelectTask} />
                  ))}
                </div>
              )}
            </Section>

            <Section title={`→ Reads Cells From (${deps.cell_deps.length})`}>
              {deps.cell_deps.length === 0 ? (
                <Empty>None</Empty>
              ) : (
                <div className="space-y-0.5">
                  {deps.cell_deps.map(([id, name, idx], i) => (
                    <div key={i} className="flex items-center gap-1 text-xs">
                      <span
                        className="cursor-pointer font-mono text-blue-400 hover:underline"
                        onClick={() => handleSelectTask(id)}
                      >
                        #{id}
                      </span>
                      <span className="truncate text-muted-foreground">{name}</span>
                      <span className="text-[10px] text-muted-foreground/60">[cell {idx}]</span>
                    </div>
                  ))}
                </div>
              )}
            </Section>

            <Section title={`← Read By (${deps.dependents.length})`}>
              {deps.dependents.length === 0 ? (
                <Empty>None</Empty>
              ) : (
                <div className="space-y-0.5">
                  {deps.dependents.map(([id, name], i) => (
                    <TaskLink key={`${id}-${i}`} taskId={id} name={name} onClick={handleSelectTask} />
                  ))}
                </div>
              )}
            </Section>
          </>
        )}
      </div>
    </ScrollArea>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="px-3 py-2">
        <CardTitle className="text-[10px] uppercase tracking-wider text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent className="px-3 pb-2 pt-0">{children}</CardContent>
    </Card>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] text-muted-foreground">{children}</div>
}

function TaskLink({ taskId, name, onClick }: { taskId: number; name: string; onClick: (id: number) => void }) {
  return (
    <div className="flex items-center gap-1 text-xs">
      <span className="cursor-pointer font-mono text-blue-400 hover:underline" onClick={() => onClick(taskId)}>
        #{taskId}
      </span>
      <span className="truncate text-muted-foreground">{name}</span>
    </div>
  )
}
