const CONTROL_BASE = '/control-api'

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
  return res.json()
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body != null ? JSON.stringify(body) : '{}',
  })
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
  return res.json()
}

async function deleteJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'DELETE' })
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
  return res.json()
}

export interface ControlStatus {
  paused: boolean
  pending_count: number
  breakpoints: BreakpointInfo[]
}

export interface BreakpointInfo {
  id: number
  pattern: string
  enabled: boolean
}

export interface PendingTaskInfo {
  task_id: number
  name: string
  hit_breakpoint: number | null
}

export interface CellInfo {
  type_name: string
  cell_index: number
  has_data: boolean
}

export interface LiveTaskInfo {
  task_id: number
  name: string
}

export interface ChildInfo {
  task_id: number
  name: string
}

export interface TaskDepsInfo {
  output_deps: [number, string][]
  cell_deps: [number, string, number][]
  dependents: [number, string][]
}

export interface DebugEvent {
  seq: number
  kind: number
  kind_name: string
  task_id: number
  detail: string
  timestamp_us: number
}

export interface SearchResult {
  task_id: number
  name: string
}

export interface GraphNode {
  task_id: number
  name: string
  state: string
}

export interface GraphEdge {
  source: number
  target: number
  edge_type: string
  label: string | null
}

export interface TaskGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  root: number
}

export interface CellDetail {
  type_name: string
  cell_index: number
  has_data: boolean
  data_preview: string | null
  data_size_bytes: number | null
}

export interface TaskStateInfo {
  task_id: number
  name: string | null
  state: string
  is_dirty: boolean
  is_in_progress: boolean
  has_output: boolean
  output_description: string | null
  cell_count: number
  child_count: number
  output_dep_count: number
  cell_dep_count: number
  dependent_count: number
  is_stateful: boolean
  is_immutable: boolean
}

export interface ActiveTask {
  task_id: number
  name: string
  state: string
  last_event_seq: number
}

export const controlApi = {
  status: () => fetchJson<ControlStatus>(`${CONTROL_BASE}/api/control/status`),
  pause: () => postJson<{ ok: boolean }>(`${CONTROL_BASE}/api/control/pause`),
  resume: () => postJson<{ ok: boolean }>(`${CONTROL_BASE}/api/control/resume`),
  step: (count = 1) => postJson<{ released: number }>(`${CONTROL_BASE}/api/control/step`, { count }),
  stepTask: (task_id: number) => postJson<{ ok: boolean }>(`${CONTROL_BASE}/api/control/step-task`, { task_id }),
  stepToIdle: (max = 100) => postJson<{ released: number }>(`${CONTROL_BASE}/api/control/step-to-idle`, { max }),
  pending: () => fetchJson<PendingTaskInfo[]>(`${CONTROL_BASE}/api/control/pending`),
  activeTasks: () => fetchJson<ActiveTask[]>(`${CONTROL_BASE}/api/control/active-tasks`),

  // Breakpoints
  listBreakpoints: () => fetchJson<BreakpointInfo[]>(`${CONTROL_BASE}/api/control/breakpoints`),
  addBreakpoint: (pattern: string) => postJson<{ id: number }>(`${CONTROL_BASE}/api/control/breakpoints`, { pattern }),
  removeBreakpoint: (id: number) => deleteJson<{ ok: boolean }>(`${CONTROL_BASE}/api/control/breakpoints/${id}`),
  toggleBreakpoint: (id: number, enabled: boolean) => postJson<{ ok: boolean }>(`${CONTROL_BASE}/api/control/breakpoints/${id}/toggle`, { enabled }),

  // Live inspection
  liveTask: (id: number) => fetchJson<LiveTaskInfo | null>(`${CONTROL_BASE}/api/live/task/${id}`),
  liveCells: (id: number) => fetchJson<CellInfo[]>(`${CONTROL_BASE}/api/live/task/${id}/cells`),
  liveChildren: (id: number) => fetchJson<ChildInfo[]>(`${CONTROL_BASE}/api/live/task/${id}/children`),
  liveDeps: (id: number) => fetchJson<TaskDepsInfo>(`${CONTROL_BASE}/api/live/task/${id}/deps`),
  taskGraph: (id: number, depth = 2) => fetchJson<TaskGraph>(`${CONTROL_BASE}/api/live/task/${id}/graph?depth=${depth}`),
  taskState: (id: number) => fetchJson<TaskStateInfo | null>(`${CONTROL_BASE}/api/live/task/${id}/state`),
  cellDetail: (id: number, cellIndex: number) => fetchJson<CellDetail | null>(`${CONTROL_BASE}/api/live/task/${id}/cells/${cellIndex}`),

  // Event log
  events: (since = 0) => fetchJson<DebugEvent[]>(`${CONTROL_BASE}/api/control/events?since=${since}`),

  // Task search
  searchTasks: (q: string, limit = 10) => fetchJson<SearchResult[]>(`${CONTROL_BASE}/api/live/search?q=${encodeURIComponent(q)}&limit=${limit}`),
}
