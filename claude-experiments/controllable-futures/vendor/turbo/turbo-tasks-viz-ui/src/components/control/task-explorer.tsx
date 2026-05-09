'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { controlApi, TaskStateInfo, CellInfo, ChildInfo, TaskDepsInfo, CellDetail } from '@/lib/control-api'
import { ScrollArea } from '@/components/ui/scroll-area'

// ─── State Config ─────────────────────────────────────────────

const STATE_CONFIG: Record<string, {
  color: string
  label: string
  dotClass: string
  pulse: boolean
}> = {
  created:     { color: '#71717a', label: 'CREATED',   dotClass: 'bg-zinc-500',    pulse: false },
  scheduled:   { color: '#fbbf24', label: 'QUEUED',    dotClass: 'bg-amber-400',   pulse: false },
  in_progress: { color: '#22d3ee', label: 'RUNNING',   dotClass: 'bg-cyan-400',    pulse: true },
  completed:   { color: '#34d399', label: 'DONE',      dotClass: 'bg-emerald-400', pulse: false },
  dirty:       { color: '#fb7185', label: 'DIRTY',     dotClass: 'bg-rose-400',    pulse: true },
}

function getState(state: string) {
  return STATE_CONFIG[state] || STATE_CONFIG.created
}

function shortName(name: string): string {
  const traitMatch = name.match(/<(\w+)\s+as\s+\w+>::(\w+)/)
  if (traitMatch) return `${traitMatch[1]}::${traitMatch[2]}`
  const parts = name.split('::')
  if (parts.length >= 2) return parts.slice(-2).join('::')
  return name
}

function typeName(fullName: string): string {
  // Extract the last segment before :: as the "module" context
  const parts = fullName.split('::')
  if (parts.length <= 1) return fullName
  // Return last 3 parts max for context
  return parts.slice(-3).join('::')
}

// ─── Types ────────────────────────────────────────────────────

interface HistoryEntry {
  taskId: number
  name: string
  state: string
}

interface ExplorerData {
  stateInfo: TaskStateInfo | null
  cells: CellInfo[]
  children: ChildInfo[]
  deps: TaskDepsInfo | null
}

// ─── Subcomponents ────────────────────────────────────────────

function StateDot({ state, size = 'sm' }: { state: string; size?: 'sm' | 'md' | 'lg' }) {
  const cfg = getState(state)
  const sizeClass = size === 'lg' ? 'h-3 w-3' : size === 'md' ? 'h-2.5 w-2.5' : 'h-2 w-2'
  return (
    <span
      className={`inline-block rounded-full ${cfg.dotClass} ${sizeClass} ${cfg.pulse ? 'animate-dot-pulse' : ''}`}
      style={{ boxShadow: `0 0 6px ${cfg.color}` }}
    />
  )
}

function Breadcrumb({
  history,
  currentName,
  onNavigate,
  onBack,
}: {
  history: HistoryEntry[]
  currentName: string
  onNavigate: (index: number) => void
  onBack: () => void
}) {
  return (
    <div className="flex items-center gap-1 animate-breadcrumb-in">
      <button
        onClick={onBack}
        className="mr-1 flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-zinc-400 transition-all hover:bg-zinc-800 hover:text-zinc-200 active:scale-95"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        BACK
      </button>
      <div className="flex items-center gap-0.5 overflow-hidden">
        {history.map((entry, i) => (
          <span key={i} className="flex shrink-0 items-center gap-0.5">
            <button
              onClick={() => onNavigate(i)}
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-300"
            >
              <StateDot state={entry.state} size="sm" />
              {shortName(entry.name)}
            </button>
            <span className="text-[10px] text-zinc-600">&rsaquo;</span>
          </span>
        ))}
        <span className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-200">
          {currentName}
        </span>
      </div>
    </div>
  )
}

function FocusedCard({
  taskId,
  name,
  stateInfo,
  childCount,
  cellCount,
  depCount,
  dependentCount,
}: {
  taskId: number
  name: string
  stateInfo: TaskStateInfo | null
  childCount: number
  cellCount: number
  depCount: number
  dependentCount: number
}) {
  const state = stateInfo?.state || 'created'
  const cfg = getState(state)

  const stats = [
    { value: childCount, label: 'children', accent: '#60a5fa' },
    { value: depCount, label: 'reads from', accent: '#c084fc' },
    { value: cellCount, label: 'cells', accent: '#34d399' },
    { value: dependentCount, label: 'read by', accent: '#fbbf24' },
  ]

  return (
    <div className={`explorer-card-glow scan-overlay relative rounded-xl bg-zinc-900/80 p-5 state-${state}`}>
      {/* Task ID + state label */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="font-mono text-[11px] text-zinc-500">#{taskId}</div>
          <div className="text-lg font-semibold leading-tight text-zinc-100">
            {shortName(name)}
          </div>
          <div className="text-[11px] text-zinc-500">{typeName(name)}</div>
        </div>
        <div className="flex items-center gap-2">
          {stateInfo?.is_dirty && (
            <span className="rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-rose-400">
              dirty
            </span>
          )}
          {stateInfo?.is_stateful && (
            <span className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-amber-400">
              stateful
            </span>
          )}
          {stateInfo?.is_immutable && (
            <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-emerald-400">
              immutable
            </span>
          )}
          <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/30 px-2.5 py-1">
            <StateDot state={state} size="md" />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: cfg.color }}>
              {cfg.label}
            </span>
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="mt-5 flex gap-2">
        {stats.map(({ value, label, accent }) => (
          <div
            key={label}
            className="flex-1 rounded-lg border border-white/[0.04] bg-black/20 px-3 py-2 text-center"
          >
            <div className="text-base font-bold" style={{ color: accent }}>{value}</div>
            <div className="text-[9px] uppercase tracking-wider text-zinc-500">{label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ExplorerSection({
  title,
  subtitle,
  count,
  accent,
  collapsed,
  onToggle,
  children,
}: {
  title: string
  subtitle?: string
  count: number | null
  accent: string
  collapsed: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="section-header flex items-center gap-2 py-1" onClick={onToggle}>
        <span
          className="text-[10px] font-bold uppercase tracking-[0.15em]"
          style={{ color: accent }}
        >
          {title}
        </span>
        {count !== null && (
          <span
            className="rounded-full px-1.5 py-0 text-[9px] font-bold"
            style={{ color: accent, background: `${accent}15` }}
          >
            {count}
          </span>
        )}
        {subtitle && (
          <span className="text-[9px] text-zinc-600">{subtitle}</span>
        )}
        <div className="section-line h-px flex-1 opacity-20" style={{ background: accent }} />
        <span
          className="text-[10px] transition-transform duration-200"
          style={{
            color: accent,
            transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
          }}
        >
          &#9660;
        </span>
      </div>
      {!collapsed && (
        <div className="mt-1.5">
          {children}
        </div>
      )}
    </div>
  )
}

function TaskPill({
  taskId,
  name,
  state,
  index,
  badge,
  onClick,
}: {
  taskId: number
  name: string
  state?: string
  index: number
  badge?: string
  onClick: () => void
}) {
  return (
    <div
      className="task-pill flex items-center gap-2 rounded-lg bg-zinc-900/50 px-3 py-2 opacity-0 animate-explorer-item-in"
      style={{ animationDelay: `${index * 30}ms` }}
      onClick={onClick}
    >
      <StateDot state={state || 'completed'} />
      <span className="font-mono text-[10px] text-zinc-500">#{taskId}</span>
      <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300">
        {shortName(name)}
      </span>
      {badge && (
        <span className="shrink-0 rounded border border-purple-500/30 bg-purple-500/10 px-1.5 py-0.5 text-[9px] text-purple-400">
          {badge}
        </span>
      )}
      <svg
        className="h-3 w-3 shrink-0 text-zinc-600"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M9 18l6-6-6-6" />
      </svg>
    </div>
  )
}

function CellRow({
  cell,
  index,
  expanded,
  detail,
  detailLoading,
  onToggle,
}: {
  cell: CellInfo
  index: number
  expanded: boolean
  detail: CellDetail | null
  detailLoading: boolean
  onToggle: () => void
}) {
  // Extract short type name from the full qualified name
  const shortType = cell.type_name.split('::').pop() || cell.type_name

  return (
    <div
      className="opacity-0 animate-explorer-item-in"
      style={{ animationDelay: `${index * 30}ms` }}
    >
      <div
        className="cell-row flex items-center gap-2 rounded-lg bg-zinc-900/50 px-3 py-2"
        onClick={onToggle}
      >
        <span
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-[9px] font-bold text-emerald-400"
          style={{ background: 'rgba(52, 211, 153, 0.1)', border: '1px solid rgba(52, 211, 153, 0.2)' }}
        >
          {cell.cell_index}
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-zinc-300">
          {shortType}
        </span>
        <span className="text-[9px] text-zinc-600">{cell.type_name !== shortType ? cell.type_name.replace(`::${shortType}`, '') : ''}</span>
        <span
          className="text-[10px] text-emerald-500 transition-transform duration-200"
          style={{ transform: expanded ? 'rotate(90deg)' : 'rotate(0)' }}
        >
          &#9654;
        </span>
      </div>
      {expanded && (
        <div className="cell-data-expand ml-3 mt-1 mb-1 rounded-lg border border-emerald-500/10 bg-black/30">
          {detailLoading ? (
            <div className="flex items-center gap-2 p-3 text-[10px] text-zinc-500">
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-emerald-500/30 border-t-emerald-400" />
              Loading cell data...
            </div>
          ) : detail?.data_preview ? (
            <div>
              {detail.data_size_bytes != null && (
                <div className="border-b border-white/[0.03] px-3 py-1.5 text-[9px] text-zinc-600">
                  {detail.data_size_bytes.toLocaleString()} bytes debug repr
                </div>
              )}
              <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all p-3 font-mono text-[10px] leading-relaxed text-zinc-300">
                {detail.data_preview}
              </pre>
            </div>
          ) : (
            <div className="p-3 text-[10px] text-zinc-600">No preview available</div>
          )}
        </div>
      )}
    </div>
  )
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-zinc-800 py-4 text-center text-[10px] text-zinc-600">
      {children}
    </div>
  )
}

function LoadingCard() {
  return (
    <div className="explorer-card-glow rounded-xl bg-zinc-900/80 p-8">
      <div className="flex items-center justify-center gap-3">
        <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-500/30 border-t-cyan-400" />
        <span className="text-xs text-zinc-500">Loading task data...</span>
      </div>
    </div>
  )
}

function EmptyExplorer() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-8">
      <div
        className="h-16 w-16 rounded-2xl border border-zinc-800 bg-zinc-900/50"
        style={{
          boxShadow: '0 0 30px rgba(34, 211, 238, 0.05), inset 0 0 30px rgba(34, 211, 238, 0.02)',
        }}
      />
      <div className="text-center">
        <div className="text-xs font-medium text-zinc-400">No task selected</div>
        <div className="mt-1 text-[10px] text-zinc-600">
          Click a task ID in the event log, search, or pending queue
        </div>
      </div>
      <div className="mt-2 flex items-center gap-4 text-[9px] text-zinc-700">
        <span>ESC to go back</span>
        <span>Click to explore</span>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────

interface Props {
  taskId: number | null
  onSelectTask: (id: number) => void
}

export function TaskExplorer({ taskId, onSelectTask }: Props) {
  const [data, setData] = useState<ExplorerData>({
    stateInfo: null,
    cells: [],
    children: [],
    deps: null,
  })
  const [loading, setLoading] = useState(true)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [expandedCell, setExpandedCell] = useState<number | null>(null)
  const [cellDetail, setCellDetail] = useState<CellDetail | null>(null)
  const [cellDetailLoading, setCellDetailLoading] = useState(false)
  const [animKey, setAnimKey] = useState(0)
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set())

  // Fetch data when taskId changes
  useEffect(() => {
    if (taskId == null) return
    setLoading(true)
    setExpandedCell(null)
    setCellDetail(null)
    setAnimKey(k => k + 1)

    Promise.all([
      controlApi.taskState(taskId).catch(() => null),
      controlApi.liveCells(taskId).catch(() => [] as CellInfo[]),
      controlApi.liveChildren(taskId).catch(() => [] as ChildInfo[]),
      controlApi.liveDeps(taskId).catch(() => null),
    ]).then(([stateInfo, cells, children, deps]) => {
      setData({ stateInfo, cells, children, deps })
      setLoading(false)
    })
  }, [taskId])

  const navigateTo = useCallback((id: number) => {
    if (taskId == null) return
    const currentName = data.stateInfo?.name || `Task #${taskId}`
    const currentState = data.stateInfo?.state || 'created'
    setHistory(prev => [...prev, { taskId, name: currentName, state: currentState }])
    onSelectTask(id)
  }, [taskId, data.stateInfo, onSelectTask])

  const navigateBack = useCallback(() => {
    if (history.length === 0) return
    const prev = history[history.length - 1]
    setHistory(h => h.slice(0, -1))
    onSelectTask(prev.taskId)
  }, [history, onSelectTask])

  const navigateToHistoryIndex = useCallback((index: number) => {
    const target = history[index]
    setHistory(h => h.slice(0, index))
    onSelectTask(target.taskId)
  }, [history, onSelectTask])

  const toggleSection = useCallback((section: string) => {
    setCollapsedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) next.delete(section)
      else next.add(section)
      return next
    })
  }, [])

  const handleExpandCell = useCallback((cellIndex: number) => {
    if (expandedCell === cellIndex) {
      setExpandedCell(null)
      setCellDetail(null)
      return
    }
    if (taskId == null) return
    setExpandedCell(cellIndex)
    setCellDetailLoading(true)
    controlApi.cellDetail(taskId, cellIndex).then(d => {
      setCellDetail(d)
      setCellDetailLoading(false)
    }).catch(() => {
      setCellDetailLoading(false)
    })
  }, [expandedCell, taskId])

  // Keyboard: Escape to go back
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        navigateBack()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [navigateBack])

  if (taskId == null) {
    return <EmptyExplorer />
  }

  const taskName = data.stateInfo?.name || `Task #${taskId}`
  const depCount = (data.deps?.output_deps.length || 0) + (data.deps?.cell_deps.length || 0)

  return (
    <ScrollArea className="h-full">
      <div
        key={animKey}
        className="animate-explorer-enter space-y-5 p-4"
      >
        {/* Breadcrumb */}
        {history.length > 0 && (
          <Breadcrumb
            history={history}
            currentName={shortName(taskName)}
            onNavigate={navigateToHistoryIndex}
            onBack={navigateBack}
          />
        )}

        {/* Focused card */}
        {loading ? (
          <LoadingCard />
        ) : (
          <FocusedCard
            taskId={taskId}
            name={taskName}
            stateInfo={data.stateInfo}
            childCount={data.children.length}
            cellCount={data.cells.length}
            depCount={depCount}
            dependentCount={data.deps?.dependents.length || 0}
          />
        )}

        {/* Output */}
        {data.stateInfo?.has_output && data.stateInfo.output_description && (
          <ExplorerSection
            title="OUTPUT"
            count={null}
            accent="#a78bfa"
            collapsed={collapsedSections.has('output')}
            onToggle={() => toggleSection('output')}
          >
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-all rounded-lg border border-violet-500/10 bg-black/30 p-3 font-mono text-[10px] leading-relaxed text-zinc-300">
              {data.stateInfo.output_description}
            </pre>
          </ExplorerSection>
        )}

        {/* Children */}
        <ExplorerSection
          title="CHILDREN"
          subtitle="tasks spawned"
          count={data.children.length}
          accent="#60a5fa"
          collapsed={collapsedSections.has('children')}
          onToggle={() => toggleSection('children')}
        >
          {data.children.length === 0 ? (
            <EmptyState>No child tasks</EmptyState>
          ) : (
            <div className="space-y-1">
              {data.children.map((child, i) => (
                <TaskPill
                  key={child.task_id}
                  taskId={child.task_id}
                  name={child.name}
                  index={i}
                  onClick={() => navigateTo(child.task_id)}
                />
              ))}
            </div>
          )}
        </ExplorerSection>

        {/* Cells */}
        <ExplorerSection
          title="CELLS"
          subtitle="stored data"
          count={data.cells.length}
          accent="#34d399"
          collapsed={collapsedSections.has('cells')}
          onToggle={() => toggleSection('cells')}
        >
          {data.cells.length === 0 ? (
            <EmptyState>No cells</EmptyState>
          ) : (
            <div className="space-y-1">
              {data.cells.map((cell, i) => (
                <CellRow
                  key={cell.cell_index}
                  cell={cell}
                  index={i}
                  expanded={expandedCell === cell.cell_index}
                  detail={expandedCell === cell.cell_index ? cellDetail : null}
                  detailLoading={expandedCell === cell.cell_index && cellDetailLoading}
                  onToggle={() => handleExpandCell(cell.cell_index)}
                />
              ))}
            </div>
          )}
        </ExplorerSection>

        {/* Dependencies */}
        {data.deps && (
          <>
            <ExplorerSection
              title="READS FROM"
              subtitle="output + cell deps"
              count={depCount}
              accent="#c084fc"
              collapsed={collapsedSections.has('deps')}
              onToggle={() => toggleSection('deps')}
            >
              {depCount === 0 ? (
                <EmptyState>No dependencies</EmptyState>
              ) : (
                <div className="space-y-1">
                  {data.deps.output_deps.map(([id, name], i) => (
                    <TaskPill
                      key={`out-${id}`}
                      taskId={id}
                      name={name}
                      index={i}
                      badge="output"
                      onClick={() => navigateTo(id)}
                    />
                  ))}
                  {data.deps.cell_deps.map(([id, name, idx], i) => (
                    <TaskPill
                      key={`cell-${id}-${idx}`}
                      taskId={id}
                      name={name}
                      index={data.deps!.output_deps.length + i}
                      badge={`cell ${idx}`}
                      onClick={() => navigateTo(id)}
                    />
                  ))}
                </div>
              )}
            </ExplorerSection>

            <ExplorerSection
              title="READ BY"
              subtitle="dependents"
              count={data.deps.dependents.length}
              accent="#fbbf24"
              collapsed={collapsedSections.has('dependents')}
              onToggle={() => toggleSection('dependents')}
            >
              {data.deps.dependents.length === 0 ? (
                <EmptyState>No dependents</EmptyState>
              ) : (
                <div className="space-y-1">
                  {data.deps.dependents.map(([id, name], i) => (
                    <TaskPill
                      key={id}
                      taskId={id}
                      name={name}
                      index={i}
                      onClick={() => navigateTo(id)}
                    />
                  ))}
                </div>
              )}
            </ExplorerSection>
          </>
        )}

        {/* Bottom padding for scroll */}
        <div className="h-4" />
      </div>
    </ScrollArea>
  )
}
