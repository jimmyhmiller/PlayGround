'use client'

import { useTaskSearch } from '@/hooks/use-task-search'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

interface Props {
  onSelect: (taskId: number) => void
}

export function TaskSearch({ onSelect }: Props) {
  const { query, setQuery, results, loading } = useTaskSearch()

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Search Tasks</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        <Input
          placeholder="Task name..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          className="h-7 text-xs"
        />
        {loading && <div className="text-xs text-muted-foreground">Searching...</div>}
        {results.length > 0 && (
          <ScrollArea className="max-h-40">
            <div className="space-y-0">
              {results.map(r => (
                <div
                  key={r.task_id}
                  onClick={() => onSelect(r.task_id)}
                  className="flex cursor-pointer gap-2 rounded px-2 py-1 text-xs hover:bg-accent"
                >
                  <span className="min-w-[40px] font-mono text-muted-foreground">#{r.task_id}</span>
                  <span className="truncate">{r.name}</span>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}
