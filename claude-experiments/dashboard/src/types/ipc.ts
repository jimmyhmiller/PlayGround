/**
 * IPC Communication Types
 *
 * Type definitions for Electron IPC channels and the preload API
 */

import type { DashboardEvent, EventFilter } from './events';
import type { CommandResult } from './state';
import type { PipelineConfig, PipelineStats, ProcessorDescriptor } from './pipeline';
import type {
  SessionNotification,
  ContentBlock,
  RequestPermissionRequest,
} from './acp';

/**
 * IPC Channel names
 */
export const IPC_CHANNELS = {
  // Legacy channels
  GET_MESSAGE: 'get-message',
  INCREMENT: 'increment',
  GET_COUNTER: 'get-counter',

  // Event channels
  EVENTS_EMIT: 'events:emit',
  EVENTS_QUERY: 'events:query',
  EVENTS_COUNT: 'events:count',
  EVENTS_PUSH: 'events:push',

  // File channels
  FILE_LOAD: 'file:load',
  FILE_WATCH: 'file:watch',
  FILE_UNWATCH: 'file:unwatch',
  FILE_GET_WATCHED: 'file:getWatched',

  // Git channels
  GIT_REFRESH: 'git:refresh',
  GIT_STATUS: 'git:status',
  GIT_DIFF: 'git:diff',
  GIT_DIFF_STAGED: 'git:diffStaged',
  GIT_START_POLLING: 'git:startPolling',
  GIT_STOP_POLLING: 'git:stopPolling',
  GIT_STAGE: 'git:stage',
  GIT_UNSTAGE: 'git:unstage',

  // State channels
  STATE_GET: 'state:get',
  STATE_COMMAND: 'state:command',

  // Eval channels
  EVAL_EXECUTE: 'eval:execute',
  EVAL_BATCH: 'eval:batch',

  // ACP channels
  ACP_SPAWN: 'acp:spawn',
  ACP_INITIALIZE: 'acp:initialize',
  ACP_NEW_SESSION: 'acp:newSession',
  ACP_RESUME_SESSION: 'acp:resumeSession',
  ACP_PROMPT: 'acp:prompt',
  ACP_CANCEL: 'acp:cancel',
  ACP_SET_MODE: 'acp:setMode',
  ACP_SHUTDOWN: 'acp:shutdown',
  ACP_IS_CONNECTED: 'acp:isConnected',
  ACP_RESPOND_PERMISSION: 'acp:respondPermission',
  ACP_SESSION_UPDATE: 'acp:sessionUpdate',
  ACP_LOAD_SESSION_HISTORY: 'acp:loadSessionHistory',
} as const;

export type IpcChannel = typeof IPC_CHANNELS[keyof typeof IPC_CHANNELS];

/**
 * Git file status
 */
export interface GitFileStatus {
  path: string;
  status: 'modified' | 'added' | 'deleted' | 'renamed' | 'untracked';
  staged: boolean;
}

/**
 * Git status response
 */
export interface GitStatus {
  branch: string;
  files: GitFileStatus[];
  ahead: number;
  behind: number;
}

/**
 * File content response
 */
export interface FileContent {
  path: string;
  content: string;
  encoding: string;
}

/**
 * Electron API exposed via preload (legacy)
 */
export interface ElectronAPI {
  getMessage(): Promise<string>;
  increment(): Promise<number>;
  getCounter(): Promise<number>;
}

/**
 * Event API exposed via preload
 */
export interface EventAPI {
  emit(type: string, payload?: unknown): Promise<DashboardEvent>;
  query(filter?: EventFilter): Promise<DashboardEvent[]>;
  count(): Promise<number>;
  subscribe(pattern: string, callback: (event: DashboardEvent) => void): () => void;
}

/**
 * File API exposed via preload
 */
export interface FileAPI {
  load(filePath: string): Promise<FileContent>;
  watch(watchPath: string): Promise<void>;
  unwatch(watchPath: string): Promise<void>;
  getWatched(): Promise<string[]>;
}

/**
 * Git API exposed via preload
 */
export interface GitAPI {
  refresh(): Promise<void>;
  status(): Promise<GitStatus>;
  diff(filePath?: string): Promise<string>;
  diffStaged(filePath?: string): Promise<string>;
  startPolling(intervalMs?: number): Promise<void>;
  stopPolling(): Promise<void>;
  stage(filePath: string): Promise<void>;
  unstage(filePath: string): Promise<void>;
}

/**
 * State API exposed via preload
 */
export interface StateAPI {
  get(path?: string): Promise<unknown>;
  command(type: string, payload?: unknown): Promise<CommandResult>;
  subscribe(path: string, callback: (event: DashboardEvent) => void): () => void;
}

/**
 * Evaluation result from the eval service
 */
export interface EvaluationResult {
  id: string;
  success: boolean;
  value?: unknown;
  displayValue: string;
  type: string;
  executionTimeMs: number;
  error?: string;
  language?: string;
}

/**
 * Evaluation request for the eval service
 */
export interface EvaluationRequest {
  id: string;
  code: string;
  language: string;
  context?: Record<string, unknown>;
  timeout?: number;
}

/**
 * Executor configuration for subprocess-based execution
 */
export interface ExecutorConfig {
  language: string;
  command: string;
  args?: string[];
  cwd?: string;
}

/**
 * Eval API exposed via preload
 */
export interface EvalAPI {
  execute(
    code: string,
    language?: string,
    context?: Record<string, unknown>
  ): Promise<EvaluationResult>;
  batch(requests: EvaluationRequest[]): Promise<EvaluationResult[]>;
  registerExecutor(config: ExecutorConfig): Promise<{ success: boolean; language: string }>;
  unregisterExecutor(language: string): Promise<{ success: boolean }>;
  getExecutors(): Promise<ExecutorConfig[]>;
}

/**
 * Shell API exposed via preload - spawn and manage processes
 */
export interface ShellAPI {
  spawn(
    id: string,
    command: string,
    args?: string[],
    options?: { cwd?: string; env?: Record<string, string> }
  ): Promise<{ success: boolean; id: string; pid?: number }>;
  kill(id: string): Promise<{ success: boolean; id?: string; error?: string }>;
  isRunning(id: string): Promise<{ running: boolean }>;
}

/**
 * Pipeline API exposed via preload - Unix-pipes style data processing
 */
export interface PipelineAPI {
  start(config: PipelineConfig): Promise<{ success: boolean; error?: string }>;
  stop(id: string): Promise<{ success: boolean; error?: string }>;
  stats(id: string): Promise<PipelineStats | undefined>;
  isRunning(id: string): Promise<{ running: boolean }>;
  list(): Promise<string[]>;
  listDetailed(): Promise<Array<{ id: string; config: PipelineConfig; stats: PipelineStats }>>;
  stopAll(): Promise<{ success: boolean }>;
  processors(): Promise<string[]>;
  describeProcessors(): Promise<ProcessorDescriptor[]>;
}

/**
 * ACP (Agent Client Protocol) API exposed via preload
 */
export interface ACPAPI {
  /** Spawn the claude-code-acp agent process */
  spawn(): Promise<void>;

  /** Initialize the ACP connection */
  initialize(): Promise<void>;

  /** Create a new session */
  newSession(cwd: string, mcpServers?: unknown[], force?: boolean): Promise<{
    sessionId: string;
    modes?: {
      availableModes: Array<{ id: string; name: string }>;
      currentModeId: string;
    };
  }>;

  /** Resume an existing session */
  resumeSession(sessionId: string, cwd: string): Promise<{
    sessionId: string;
    modes?: { availableModes: Array<{ id: string; name: string }>; currentModeId: string };
  }>;

  /** Send a prompt to the agent */
  prompt(sessionId: string, content: string | ContentBlock[]): Promise<{ stopReason: string }>;

  /** Cancel an ongoing prompt */
  cancel(sessionId: string): Promise<void>;

  /** Set session mode (e.g., 'plan', 'act') */
  setMode(sessionId: string, modeId: string): Promise<void>;

  /** Shutdown the agent connection */
  shutdown(): Promise<void>;

  /** Check if connected */
  isConnected(): Promise<boolean>;

  /** Respond to a permission request with the selected optionId */
  respondToPermission(requestId: string, optionId: string): Promise<void>;

  /** Subscribe to session updates */
  subscribeUpdates(callback: (update: SessionNotification) => void): () => void;

  /** Subscribe to permission requests */
  subscribePermissions(callback: (request: RequestPermissionRequest) => void): () => void;

  /** Load session history from Claude's local files */
  loadSessionHistory(sessionId: string, cwd: string): Promise<Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>>;
}

/**
 * Complete Window API (extends global Window)
 */
declare global {
  interface Window {
    electronAPI: ElectronAPI;
    eventAPI: EventAPI;
    fileAPI: FileAPI;
    gitAPI: GitAPI;
    stateAPI: StateAPI;
    evalAPI: EvalAPI;
    shellAPI: ShellAPI;
    pipelineAPI: PipelineAPI;
    acpAPI: ACPAPI;
  }
}

export {};
