/**
 * Pipeline Module
 *
 * Unix-pipes style data pipeline system.
 * Re-exports all pipeline components.
 */

export * from './ProcessorRegistry';
export * from './PipelineRuntime';
export * from './PipelineService';

// Re-export types
export type {
  Processor,
  ProcessorInstance,
  ProcessorConfig,
  ProcessorContext,
  ProcessorDescriptor,
  PipelineConfig,
  PipelineStage,
  PipelineInstance,
  PipelineStats,
  PipelineErrorStrategy,
  FileDropPayload,
  FileChangePayload,
} from '../../types/pipeline';
