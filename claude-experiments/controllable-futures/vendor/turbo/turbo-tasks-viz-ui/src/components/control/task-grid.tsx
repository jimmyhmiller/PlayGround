'use client'

import { useRef, useEffect, useState } from 'react'
import { ActiveTask } from '@/lib/control-api'
import { ScrollArea } from '@/components/ui/scroll-area'

interface Props {
  tasks: ActiveTask[]
  selectedTaskId: number | null
  onSelectTask: (taskId: number) => void
  pendingTaskIds: Set<number>
}

const STATE_COLORS: Record<string, string> = {
  created: 'bg-zinc-700',
  scheduled: 'bg-yellow-500',
  in_progress: 'bg-blue-500',
  completed: 'bg-green-600',
  dirty: 'bg-red-500',
}

export function TaskGrid({ tasks, selectedTaskId, onSelectTask, pendingTaskIds }: Props) {
  const [visible, setVisible] = useState(true)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [tasks.length])

  if (!visible) {
    return (
      <div className="flex items-center border-b px-3 py-1">
        <button
          className="text-[10px] text-muted-foreground hover:text-zinc-300"
          onClick={() => setVisible(true)}
        >
          Show task grid ({tasks.length} tasks)
        </button>
      </div>
    )
  }

  return (
    <div className="border-b">
      <div className="flex items-center justify-between px-3 py-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Task Grid ({tasks.length})
        </span>
        <button
          className="text-[10px] text-muted-foreground hover:text-zinc-300"
          onClick={() => setVisible(false)}
        >
          Hide
        </button>
      </div>
      <ScrollArea className="max-h-24 px-3 pb-2">
        <div className="flex flex-wrap gap-[2px]">
          {tasks.map((task) => (
            <div
              key={task.task_id}
              className={`
                h-3 w-3 cursor-pointer rounded-[2px]
                ${STATE_COLORS[task.state] || STATE_COLORS.created}
                ${selectedTaskId === task.task_id ? 'ring-1 ring-white' : ''}
                ${pendingTaskIds.has(task.task_id) ? 'animate-pulse' : ''}
              `}
              title={`#${task.task_id} ${task.name} (${task.state})`}
              onClick={() => onSelectTask(task.task_id)}
            />
          ))}
          <div ref={endRef} />
        </div>
      </ScrollArea>
      <div className="flex items-center gap-3 px-3 pb-1">
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-2 rounded-[1px] bg-zinc-700" /> created
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-2 rounded-[1px] bg-yellow-500" /> scheduled
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-2 rounded-[1px] bg-blue-500" /> running
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-2 rounded-[1px] bg-green-600" /> completed
        </span>
        <span className="flex items-center gap-1 text-[9px] text-muted-foreground">
          <span className="inline-block h-2 w-2 rounded-[1px] bg-red-500" /> dirty
        </span>
      </div>
    </div>
  )
}
