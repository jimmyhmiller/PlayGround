/**
 * Pipeline Type Definitions
 *
 * Unix-pipes style data pipeline system where sources emit data,
 * processors transform it, and sinks display it - all connected
 * via named event channels.
 */

// ========== Processor Types ==========

/**
 * Context provided to processor instances during execution
 */
export interface ProcessorContext {
  /** Emit events to arbitrary channels (for multi-output processors) */
  emit: (type: string, payload: unknown) => void;

  /** Pipeline scope for namespacing internal channels */
  scope: string;

  /** Pipeline ID this processor belongs to */
  pipelineId: string;

  /** Stage index within the pipeline */
  stageIndex: number;
}

/**
 * A processor instance created by a Processor factory.
 * Handles actual data processing.
 */
export interface ProcessorInstance {
  /**
   * Process a single input value.
   * @returns Transformed value, array (each emitted separately), or undefined (filter out)
   */
  process: (input: unknown) => Promise<unknown | unknown[] | undefined>;

  /**
   * Optional cleanup when pipeline is destroyed
   */
  destroy?: () => void;

  /**
   * Optional state getter for debugging/inspection
   */
  getState?: () => unknown;
}

/**
 * Configuration for a processor instance
 */
export interface ProcessorConfig {
  /** Processor name from registry */
  processor: string;

  /** Processor-specific configuration */
  config?: Record<string, unknown>;
}

/**
 * A Processor definition - factory for creating processor instances.
 * Registered by name in the ProcessorRegistry.
 */
export interface Processor {
  /** Unique name for this processor type */
  name: string;

  /** Human-readable description (for LLM and docs) */
  description: string;

  /** JSON schema for processor configuration (for validation/LLM hints) */
  configSchema?: Record<string, unknown>;

  /**
   * Create a processor instance.
   * Called once per pipeline stage.
   */
  create: (config: ProcessorConfig, context: ProcessorContext) => ProcessorInstance;
}

// ========== Pipeline Types ==========

/**
 * A single stage in a pipeline
 */
export interface PipelineStage {
  /** Optional stage ID (auto-generated if not provided) */
  id?: string;

  /** Processor name from registry */
  processor: string;

  /** Processor-specific configuration */
  config?: Record<string, unknown>;
}

/**
 * Error handling strategy for pipelines
 */
export type PipelineErrorStrategy =
  | 'skip'   // Skip failed items, continue processing (default)
  | 'emit'   // Emit error events to {sink}.error channel
  | 'stop';  // Stop the pipeline on error

/**
 * Full pipeline configuration
 */
export interface PipelineConfig {
  /** Unique pipeline ID */
  id: string;

  /** Human-readable name (optional) */
  name?: string;

  /** Description (optional) */
  description?: string;

  /** Event pattern to subscribe to as source */
  source: string;

  /** Event type to emit final output */
  sink: string;

  /** Ordered list of processing stages */
  stages: PipelineStage[];

  /** Error handling strategy (default: 'skip') */
  onError?: PipelineErrorStrategy;
}

// ========== Runtime Types ==========

/**
 * Statistics for a running pipeline
 */
export interface PipelineStats {
  /** Number of inputs received */
  inputCount: number;

  /** Number of outputs emitted */
  outputCount: number;

  /** Number of errors encountered */
  errorCount: number;

  /** Timestamp of last input */
  lastInput: number;

  /** Timestamp of last output */
  lastOutput: number;
}

/**
 * Pipeline instance state
 */
export interface PipelineInstance {
  /** Pipeline ID */
  id: string;

  /** Original configuration */
  config: PipelineConfig;

  /** Processor instances for each stage */
  stages: ProcessorInstance[];

  /** Unsubscribe function for source events */
  unsubscribe: () => void;

  /** Runtime statistics */
  stats: PipelineStats;
}

/**
 * Processor descriptor for LLM/API discovery
 */
export interface ProcessorDescriptor {
  name: string;
  description: string;
  configSchema?: Record<string, unknown>;
}

// ========== Event Types ==========

/**
 * Events emitted by the pipeline system
 */
export interface PipelineEvents {
  'pipeline.started': {
    id: string;
    name?: string;
    source: string;
    sink: string;
    stages: string[];
  };

  'pipeline.stopped': {
    id: string;
    stats: PipelineStats;
  };

  'pipeline.error': {
    pipelineId: string;
    stageIndex: number;
    processor: string;
    error: string;
    input: unknown;
  };
}

// ========== Source Widget Types ==========

/**
 * File drop event payload
 */
export interface FileDropPayload {
  /** Full file path */
  filePath: string;

  /** File name only */
  fileName: string;

  /** File content as string */
  content: string;

  /** MIME type if available */
  type?: string;

  /** File size in bytes */
  size: number;
}

/**
 * File change event payload
 */
export interface FileChangePayload {
  /** File path that changed */
  filePath: string;

  /** New file content */
  content: string;

  /** Type of change */
  changeType: 'change' | 'add' | 'unlink';

  /** Timestamp of change */
  timestamp: number;
}
