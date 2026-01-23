// Trace event types matching the Rust evaluator output

export interface BindingCreatedEvent {
  type: 'binding_created';
  name: string;
  value: string;
  is_static: boolean;
  cause: string | null;
}

export interface BindingUpdatedEvent {
  type: 'binding_updated';
  name: string;
  old: string;
  new: string;
  was_static: boolean;
  is_static: boolean;
}

export interface FunctionEnterEvent {
  type: 'function_enter';
  name: string;
  args: [string, boolean][]; // [repr, is_static]
}

export interface FunctionExitEvent {
  type: 'function_exit';
  name: string;
  result: string;
  is_static: boolean;
}

export interface LoopIterationEvent {
  type: 'loop_iteration';
  loop_type: string;
  iteration: number;
  condition: string;
  condition_static: boolean;
}

export interface BecameDynamicEvent {
  type: 'became_dynamic';
  expr: string;
  reason: string;
}

export interface BailedOutEvent {
  type: 'bailed_out';
  reason: string;
  context: string;
}

export interface EmittedResidualEvent {
  type: 'emitted_residual';
  construct: string;      // "while_loop", "for_loop", "function_call", "try_catch", "if_statement", "switch"
  reason: string;         // "dynamic_condition", "depth_limit", "iteration_limit", etc.
  residual_preview: string;  // truncated preview of what was emitted
}

export type TraceEvent =
  | BindingCreatedEvent
  | BindingUpdatedEvent
  | FunctionEnterEvent
  | FunctionExitEvent
  | LoopIterationEvent
  | BecameDynamicEvent
  | BailedOutEvent
  | EmittedResidualEvent;

// Source location information
export interface SourceLocation {
  line: number;
  column: number;
  start: number;  // byte offset
  end: number;    // byte offset
}

// Binding snapshot from the trace
export interface BindingSnapshot {
  name: string;
  value: string;
  is_static: boolean;
  scope: number;
}

export interface TraceEntry {
  seq: number;
  depth: number;
  stack: string[];
  event: TraceEvent;
  location?: SourceLocation;  // source location if available
  env?: BindingSnapshot[];     // environment snapshot
}

export interface Trace {
  source?: string;  // original source code
  events: TraceEntry[];
}

// Computed binding state at a point in time (for computed view)
export interface BindingState {
  name: string;
  value: string;
  isStatic: boolean;
  lastUpdatedAt: number; // seq number
}
