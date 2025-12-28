/**
 * IPC Communication Types
 *
 * Type definitions for Electron IPC channels and the preload API
 */

import type { DashboardEvent, EventFilter } from './events';
import type { CommandResult } from './state';

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
  EVAL_BENCHMARK: 'eval:benchmark',
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
}

/**
 * Evaluation request for the eval service
 */
export interface EvaluationRequest {
  id: string;
  code: string;
  language: 'javascript' | 'typescript';
  context?: Record<string, unknown>;
  timeout?: number;
}

/**
 * Benchmark result from the eval service
 */
export interface BenchmarkResult {
  runs: Array<{ iteration: number; executionTimeMs: number; success: boolean }>;
  stats: { mean: number; median: number; stdDev: number; min: number; max: number };
}

/**
 * Eval API exposed via preload
 */
export interface EvalAPI {
  execute(
    code: string,
    language?: 'javascript' | 'typescript',
    context?: Record<string, unknown>
  ): Promise<EvaluationResult>;
  batch(requests: EvaluationRequest[]): Promise<EvaluationResult[]>;
  benchmark(
    code: string,
    iterations?: number,
    warmupIterations?: number,
    context?: Record<string, unknown>
  ): Promise<BenchmarkResult>;
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
  }
}

export {};
