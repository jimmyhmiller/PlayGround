'use client'

import { useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import { Panel, Group, Separator } from 'react-resizable-panels'
import { useControlStatus } from '@/hooks/use-control-status'
import { useEventLog } from '@/hooks/use-event-log'
import { useActiveTasks } from '@/hooks/use-active-tasks'
import { controlApi } from '@/lib/control-api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Separator as UISeparator } from '@/components/ui/separator'
import { StatusBadge } from './status-badge'
import { Breakpoints } from './breakpoints'
import { PendingQueue } from './pending-queue'
import { TaskSearch } from './task-search'
import { EventLog } from './event-log'
import { TaskInspector } from './task-inspector'
import { SpaceExplorer } from './space-explorer'
import { TaskGrid } from './task-grid'
import { Pause, Play, SkipForward, RotateCcw } from 'lucide-react'

const TaskGraphView = dynamic(
  () => import('./task-graph').then(mod => mod.TaskGraphView),
  { ssr: false, loading: () => <div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading graph...</div> },
)

export function ControlView() {
  const {
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
  } = useControlStatus()

  const { events, clearEvents } = useEventLog()
  const { tasks: activeTasks } = useActiveTasks()
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null)
  const [stepCount, setStepCount] = useState(1)
  const [mainPanelMode, setMainPanelMode] = useState<'inspector' | 'graph' | 'explorer'>('inspector')

  const pendingTaskIds = useMemo(() => new Set(pending.map(p => p.task_id)), [pending])

  if (error) {
    return (
      <div className="p-6">
        <div className="text-destructive">
          <strong>Control center unavailable:</strong> {error}
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Make sure TURBO_TASKS_VIZ=1 is set and the process is running with the visualizer feature enabled.
        </p>
      </div>
    )
  }

  if (!status) {
    return <div className="p-6 text-muted-foreground">Connecting to control center...</div>
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Top control bar */}
      <div className="flex shrink-0 items-center gap-2 border-b px-4 py-2">
        <StatusBadge paused={status.paused} />
        {status.paused ? (
          <Button size="xs" onClick={resume} className="bg-green-700 hover:bg-green-600">
            <Play className="mr-1 h-3 w-3" /> Resume
          </Button>
        ) : (
          <Button size="xs" onClick={pause} className="bg-red-700 hover:bg-red-600">
            <Pause className="mr-1 h-3 w-3" /> Pause
          </Button>
        )}
        <Button size="xs" variant="secondary" onClick={() => step(stepCount)} disabled={!status.paused}>
          <SkipForward className="mr-1 h-3 w-3" /> Step
        </Button>
        <Input
          type="number"
          min={1}
          value={stepCount}
          onChange={e => setStepCount(Math.max(1, parseInt(e.target.value) || 1))}
          className="h-7 w-14 text-center text-xs"
        />
        <Button
          size="xs"
          variant="outline"
          onClick={async () => {
            await controlApi.stepToIdle(100)
          }}
          disabled={!status.paused}
        >
          Step to Idle
        </Button>
        <span className="text-xs text-muted-foreground">{status.pending_count} pending</span>
        <div className="flex-1" />
        <Button size="xs" variant="ghost" onClick={clearEvents} title="Clear event log">
          <RotateCcw className="h-3 w-3" />
        </Button>
      </div>

      {/* Task grid strip */}
      <TaskGrid
        tasks={activeTasks}
        selectedTaskId={selectedTaskId}
        onSelectTask={setSelectedTaskId}
        pendingTaskIds={pendingTaskIds}
      />

      {/* Main resizable layout */}
      <div className="min-h-0 flex-1 overflow-hidden">
        <Group orientation="horizontal" id="control-h" style={{ height: '100%' }}>
          {/* Left sidebar */}
          <Panel id="sidebar" defaultSize="18" minSize="10" maxSize="35">
            <div className="flex h-full flex-col gap-2 overflow-auto border-r p-2">
              <Breakpoints
                breakpoints={status.breakpoints}
                onAdd={addBreakpoint}
                onToggle={toggleBreakpoint}
                onRemove={removeBreakpoint}
              />
              <UISeparator />
              <PendingQueue
                pending={pending}
                selectedTaskId={selectedTaskId}
                onSelect={setSelectedTaskId}
                onRelease={stepTask}
                paused={status.paused}
              />
              <UISeparator />
              <TaskSearch onSelect={setSelectedTaskId} />
            </div>
          </Panel>

          <Separator className="w-1.5 bg-transparent hover:bg-blue-500/20 active:bg-blue-500/30 transition-colors cursor-col-resize" />

          {/* Main content area */}
          <Panel id="main" defaultSize="82" minSize="40">
            <Group orientation="vertical" id="control-v" style={{ height: '100%' }}>
              {/* Inspector / Graph — main focus */}
              <Panel id="inspector" defaultSize="70" minSize="20">
                <div className="flex h-full flex-col overflow-hidden">
                  {selectedTaskId != null || mainPanelMode === 'explorer' ? (
                    <>
                      <div className="flex shrink-0 items-center gap-1 border-b px-3 py-1">
                        {selectedTaskId != null && (
                          <span className="mr-2 text-xs font-semibold text-muted-foreground">Task #{selectedTaskId}</span>
                        )}
                        <Button
                          variant={mainPanelMode === 'inspector' ? 'default' : 'ghost'}
                          size="sm"
                          className="h-6 text-[10px]"
                          onClick={() => setMainPanelMode('inspector')}
                        >
                          Inspector
                        </Button>
                        <Button
                          variant={mainPanelMode === 'graph' ? 'default' : 'ghost'}
                          size="sm"
                          className="h-6 text-[10px]"
                          onClick={() => setMainPanelMode('graph')}
                        >
                          Graph
                        </Button>
                        <Button
                          variant={mainPanelMode === 'explorer' ? 'default' : 'ghost'}
                          size="sm"
                          className="h-6 text-[10px]"
                          onClick={() => setMainPanelMode('explorer')}
                        >
                          Explorer
                        </Button>
                      </div>
                      <div className="min-h-0 flex-1 overflow-hidden">
                        {mainPanelMode === 'inspector' && selectedTaskId != null ? (
                          <TaskInspector taskId={selectedTaskId} onSelectTask={setSelectedTaskId} />
                        ) : mainPanelMode === 'graph' && selectedTaskId != null ? (
                          <TaskGraphView taskId={selectedTaskId} onSelectTask={(id) => { setSelectedTaskId(id) }} />
                        ) : mainPanelMode === 'explorer' ? (
                          <SpaceExplorer taskId={selectedTaskId} onSelectTask={setSelectedTaskId} />
                        ) : (
                          <div className="flex h-full items-center justify-center p-4 text-center text-xs text-muted-foreground">
                            Click a task ID in the event log, pending queue, or search results to inspect it
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="flex h-full items-center justify-center p-4 text-center text-xs text-muted-foreground">
                      Click a task ID in the event log, pending queue, or search results to inspect it
                    </div>
                  )}
                </div>
              </Panel>

              <Separator className="h-1.5 bg-transparent hover:bg-blue-500/20 active:bg-blue-500/30 transition-colors cursor-row-resize" />

              {/* Event log — bottom panel */}
              <Panel id="events" defaultSize="30" minSize="8" maxSize="60">
                <EventLog events={events} onSelectTask={setSelectedTaskId} />
              </Panel>
            </Group>
          </Panel>
        </Group>
      </div>
    </div>
  )
}
