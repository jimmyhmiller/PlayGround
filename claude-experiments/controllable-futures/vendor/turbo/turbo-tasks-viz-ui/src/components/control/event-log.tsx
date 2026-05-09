'use client'

import { useEffect, useRef } from 'react'
import { DebugEvent } from '@/lib/control-api'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { formatDuration, EVENT_KIND_COLORS } from '@/lib/utils'

interface Props {
  events: DebugEvent[]
  onSelectTask: (taskId: number) => void
}

export function EventLog({ events, onSelectTask }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  // Count step markers up to each index for alternating group shading
  const stepIndices: number[] = []
  let stepCount = 0
  for (let i = 0; i < events.length; i++) {
    if (events[i].kind === 255 || events[i].kind === 254) {
      stepCount++
    }
    stepIndices.push(stepCount)
  }

  return (
    <div className="flex h-full flex-col border-t">
      <div className="shrink-0 border-b px-3 py-1.5">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Event Log</span>
        <span className="ml-2 text-xs text-muted-foreground">({events.length})</span>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-0 p-1">
          {events.length === 0 ? (
            <div className="p-4 text-center text-xs text-muted-foreground">
              No events yet. Events stream when tasks are running.
            </div>
          ) : (
            events.map((ev, i) => {
              // Step marker — render as divider
              if (ev.kind === 255) {
                return (
                  <div key={ev.seq} className="flex items-center gap-2 px-3 py-1.5">
                    <div className="h-px flex-1 bg-amber-600/50" />
                    <span className="text-[10px] font-semibold text-amber-500">
                      {ev.detail}
                    </span>
                    <div className="h-px flex-1 bg-amber-600/50" />
                  </div>
                )
              }

              // Resume marker — render as divider
              if (ev.kind === 254) {
                return (
                  <div key={ev.seq} className="flex items-center gap-2 px-3 py-1.5">
                    <div className="h-px flex-1 bg-green-600/50" />
                    <span className="text-[10px] font-semibold text-green-500">
                      {ev.detail}
                    </span>
                    <div className="h-px flex-1 bg-green-600/50" />
                  </div>
                )
              }

              const isOddGroup = stepIndices[i] % 2 === 1

              return (
                <div
                  key={ev.seq}
                  className={`flex items-center gap-2 rounded px-2 py-0.5 text-xs hover:bg-accent/50 ${isOddGroup ? 'bg-zinc-900/30' : ''}`}
                >
                  <span className="w-16 shrink-0 font-mono text-[10px] text-muted-foreground">
                    {formatDuration(ev.timestamp_us)}
                  </span>
                  <Badge
                    className={`${EVENT_KIND_COLORS[ev.kind_name] || 'bg-zinc-600'} h-5 shrink-0 px-1.5 text-[10px] text-white`}
                  >
                    {ev.kind_name}
                  </Badge>
                  <span
                    className="shrink-0 cursor-pointer font-mono text-blue-400 hover:underline"
                    onClick={() => onSelectTask(ev.task_id)}
                  >
                    #{ev.task_id}
                  </span>
                  {ev.detail && (
                    <span className="min-w-0 truncate text-muted-foreground" title={ev.detail}>
                      {ev.detail}
                    </span>
                  )}
                </div>
              )
            })
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  )
}
