'use client'

import { PendingTaskInfo } from '@/lib/control-api'
import { middleTruncate } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

interface Props {
  pending: PendingTaskInfo[]
  selectedTaskId: number | null
  onSelect: (taskId: number) => void
  onRelease: (taskId: number) => void
  paused: boolean
}

export function PendingQueue({ pending, selectedTaskId, onSelect, onRelease, paused }: Props) {
  return (
    <Card className="flex flex-1 flex-col overflow-hidden">
      <CardHeader className="shrink-0 pb-2">
        <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">
          Pending Tasks ({pending.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        {pending.length === 0 ? (
          <div className="px-4 py-2 text-xs text-muted-foreground">
            {paused ? 'No tasks pending' : 'Not paused — tasks are scheduling freely'}
          </div>
        ) : (
          <ScrollArea className="h-full">
            <div className="space-y-0">
              {pending.map(t => (
                <div
                  key={t.task_id}
                  onClick={() => onSelect(t.task_id)}
                  className={`flex cursor-pointer items-center gap-2 border-b border-border/30 px-3 py-1.5 text-xs hover:bg-accent/50 ${
                    selectedTaskId === t.task_id ? 'bg-blue-500/20' : t.hit_breakpoint != null ? 'bg-red-500/10' : ''
                  }`}
                >
                  <span className="min-w-[40px] font-mono text-muted-foreground">#{t.task_id}</span>
                  <span className="min-w-0 flex-1 text-[10px]" title={t.name}>{middleTruncate(t.name, 40)}</span>
                  {t.hit_breakpoint != null && (
                    <span className="shrink-0 font-semibold text-destructive">BP#{t.hit_breakpoint}</span>
                  )}
                  <Button
                    size="xs"
                    variant="outline"
                    className="h-5 shrink-0 px-1.5 text-[10px]"
                    onClick={e => { e.stopPropagation(); onRelease(t.task_id) }}
                  >
                    Release
                  </Button>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}
