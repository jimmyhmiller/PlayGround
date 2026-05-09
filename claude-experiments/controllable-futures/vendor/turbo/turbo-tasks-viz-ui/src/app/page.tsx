'use client'

import { ControlView } from '@/components/control/control-view'

export default function Page() {
  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b bg-zinc-950 px-4 py-2">
        <span className="text-sm font-bold text-zinc-100">turbo-tasks debugger</span>
      </header>
      <main className="flex-1 overflow-hidden">
        <ControlView />
      </main>
    </div>
  )
}
