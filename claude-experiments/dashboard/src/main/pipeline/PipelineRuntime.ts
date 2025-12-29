/**
 * PipelineRuntime
 *
 * Manages running pipelines - creates instances, connects them to events,
 * and handles the flow of data through processor stages.
 */

import type {
  PipelineConfig,
  PipelineInstance,
  PipelineStats,
  ProcessorInstance,
  ProcessorContext,
} from '../../types/pipeline';
import { getProcessorRegistry } from './ProcessorRegistry';

/**
 * Event emitter interface (matches our EventStore)
 */
interface EventEmitter {
  emit(type: string, payload: unknown): void;
  subscribe(pattern: string, callback: (event: { type: string; payload: unknown }) => void): () => void;
}

/**
 * Pipeline Runtime
 *
 * Executes pipelines by:
 * 1. Subscribing to source events
 * 2. Processing through each stage
 * 3. Emitting to sink channel
 */
export class PipelineRuntime {
  private events: EventEmitter;
  private pipelines: Map<string, PipelineInstance> = new Map();
  private scope: string;

  constructor(events: EventEmitter, scope: string = 'pipeline') {
    this.events = events;
    this.scope = scope;
  }

  /**
   * Create and start a pipeline from configuration
   */
  start(config: PipelineConfig): { success: boolean; error?: string } {
    if (this.pipelines.has(config.id)) {
      return { success: false, error: `Pipeline already running: ${config.id}` };
    }

    const registry = getProcessorRegistry();

    // Validate all processors exist before starting
    for (const stage of config.stages) {
      if (!registry.has(stage.processor)) {
        return { success: false, error: `Unknown processor: ${stage.processor}` };
      }
    }

    try {
      // Create processor instances for each stage
      const stages: ProcessorInstance[] = [];
      for (let i = 0; i < config.stages.length; i++) {
        const stageConfig = config.stages[i];
        const context = this.createContext(config, i);
        const instance = registry.createInstance(
          {
            processor: stageConfig.processor,
            config: stageConfig.config,
          },
          context
        );
        stages.push(instance);
      }

      const stats: PipelineStats = {
        inputCount: 0,
        outputCount: 0,
        errorCount: 0,
        lastInput: 0,
        lastOutput: 0,
      };

      // Subscribe to source events
      const unsubscribe = this.events.subscribe(config.source, async (event) => {
        await this.processEvent(config, stages, stats, event.payload);
      });

      const instance: PipelineInstance = {
        id: config.id,
        config,
        stages,
        unsubscribe,
        stats,
      };

      this.pipelines.set(config.id, instance);

      this.events.emit('pipeline.started', {
        id: config.id,
        name: config.name,
        source: config.source,
        sink: config.sink,
        stages: config.stages.map(s => s.processor),
      });

      console.log(`[PipelineRuntime] Started pipeline: ${config.id}`);
      return { success: true };
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      console.error(`[PipelineRuntime] Failed to start pipeline ${config.id}:`, error);
      return { success: false, error };
    }
  }

  /**
   * Process a single event through the pipeline
   */
  private async processEvent(
    config: PipelineConfig,
    stages: ProcessorInstance[],
    stats: PipelineStats,
    input: unknown
  ): Promise<void> {
    stats.inputCount++;
    stats.lastInput = Date.now();

    try {
      // Process through each stage
      let data: unknown | unknown[] = input;

      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];

        // Handle array input (from flatten or previous multi-output)
        if (Array.isArray(data)) {
          // Process each item, collect results
          const results: unknown[] = [];
          for (const item of data) {
            const result = await stage.process(item);
            if (result !== undefined) {
              if (Array.isArray(result)) {
                results.push(...result);
              } else {
                results.push(result);
              }
            }
          }
          data = results.length > 0 ? results : undefined;
        } else {
          // Single item processing
          const result = await stage.process(data);
          data = result;
        }

        // If filtered out, stop processing
        if (data === undefined) {
          return;
        }
      }

      // Emit final result(s)
      if (Array.isArray(data)) {
        for (const item of data) {
          this.emitOutput(config, stats, item);
        }
      } else if (data !== undefined) {
        this.emitOutput(config, stats, data);
      }
    } catch (err) {
      stats.errorCount++;
      const error = err instanceof Error ? err.message : String(err);

      if (config.onError === 'emit') {
        this.events.emit(`${config.sink}.error`, {
          pipelineId: config.id,
          error,
          input,
        });
      } else if (config.onError === 'stop') {
        this.stop(config.id);
      }
      // 'skip' (default): just log and continue
      console.error(`[Pipeline ${config.id}] Error:`, error);
    }
  }

  /**
   * Emit output to sink channel
   */
  private emitOutput(config: PipelineConfig, stats: PipelineStats, data: unknown): void {
    stats.outputCount++;
    stats.lastOutput = Date.now();
    this.events.emit(config.sink, data);
  }

  /**
   * Create processor context
   */
  private createContext(config: PipelineConfig, stageIndex: number): ProcessorContext {
    return {
      emit: (type, payload) => this.events.emit(type, payload),
      scope: `${this.scope}.${config.id}`,
      pipelineId: config.id,
      stageIndex,
    };
  }

  /**
   * Stop a running pipeline
   */
  stop(id: string): { success: boolean; error?: string } {
    const pipeline = this.pipelines.get(id);
    if (!pipeline) {
      return { success: false, error: `Pipeline not found: ${id}` };
    }

    // Unsubscribe from source
    pipeline.unsubscribe();

    // Destroy all processor instances
    for (const stage of pipeline.stages) {
      stage.destroy?.();
    }

    this.pipelines.delete(id);

    this.events.emit('pipeline.stopped', {
      id,
      stats: pipeline.stats,
    });

    console.log(`[PipelineRuntime] Stopped pipeline: ${id}`);
    return { success: true };
  }

  /**
   * Get pipeline stats
   */
  getStats(id: string): PipelineStats | undefined {
    return this.pipelines.get(id)?.stats;
  }

  /**
   * Get pipeline configuration
   */
  getConfig(id: string): PipelineConfig | undefined {
    return this.pipelines.get(id)?.config;
  }

  /**
   * Check if a pipeline is running
   */
  isRunning(id: string): boolean {
    return this.pipelines.has(id);
  }

  /**
   * List running pipeline IDs
   */
  list(): string[] {
    return Array.from(this.pipelines.keys());
  }

  /**
   * Get detailed info for all running pipelines
   */
  listDetailed(): Array<{ id: string; config: PipelineConfig; stats: PipelineStats }> {
    return Array.from(this.pipelines.values()).map(p => ({
      id: p.id,
      config: p.config,
      stats: p.stats,
    }));
  }

  /**
   * Stop all pipelines
   */
  stopAll(): void {
    for (const id of this.pipelines.keys()) {
      this.stop(id);
    }
  }
}
