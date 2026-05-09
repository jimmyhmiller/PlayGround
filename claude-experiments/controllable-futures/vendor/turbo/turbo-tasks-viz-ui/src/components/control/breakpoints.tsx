'use client'

import { useState } from 'react'
import { BreakpointInfo } from '@/lib/control-api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { X } from 'lucide-react'

interface Props {
  breakpoints: BreakpointInfo[]
  onAdd: (pattern: string) => void
  onToggle: (id: number, enabled: boolean) => void
  onRemove: (id: number) => void
}

export function Breakpoints({ breakpoints, onAdd, onToggle, onRemove }: Props) {
  const [input, setInput] = useState('')

  const handleAdd = () => {
    if (!input.trim()) return
    onAdd(input.trim())
    setInput('')
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Breakpoints</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex gap-1.5">
          <Input
            placeholder="Task name pattern..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
            className="h-7 text-xs"
          />
          <Button onClick={handleAdd} size="xs" variant="secondary">Add</Button>
        </div>
        {breakpoints.length === 0 ? (
          <div className="text-xs text-muted-foreground">No breakpoints set</div>
        ) : (
          <div className="space-y-1">
            {breakpoints.map(bp => (
              <div key={bp.id} className="flex items-center gap-2 text-xs">
                <Checkbox
                  checked={bp.enabled}
                  onCheckedChange={() => onToggle(bp.id, !bp.enabled)}
                />
                <code className={`flex-1 ${bp.enabled ? '' : 'text-muted-foreground line-through'}`}>
                  {bp.pattern}
                </code>
                <button
                  onClick={() => onRemove(bp.id)}
                  className="text-destructive hover:text-destructive/80"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
