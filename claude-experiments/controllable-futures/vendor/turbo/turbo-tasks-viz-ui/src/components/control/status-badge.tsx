'use client'

import { Badge } from '@/components/ui/badge'

export function StatusBadge({ paused }: { paused: boolean }) {
  return (
    <Badge className={paused ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}>
      {paused ? 'PAUSED' : 'RUNNING'}
    </Badge>
  )
}
