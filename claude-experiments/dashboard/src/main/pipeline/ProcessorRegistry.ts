/**
 * ProcessorRegistry
 *
 * Central registry for named processors.
 * Processors are registered by name and looked up for pipeline construction.
 */

import type {
  Processor,
  ProcessorConfig,
  ProcessorContext,
  ProcessorInstance,
  ProcessorDescriptor,
} from '../../types/pipeline';

/**
 * Registry for named processors.
 */
export class ProcessorRegistry {
  private processors: Map<string, Processor> = new Map();

  /**
   * Register a processor
   */
  register(processor: Processor): void {
    if (this.processors.has(processor.name)) {
      console.warn(`[ProcessorRegistry] Overwriting processor: ${processor.name}`);
    }
    this.processors.set(processor.name, processor);
    console.log(`[ProcessorRegistry] Registered: ${processor.name}`);
  }

  /**
   * Register multiple processors
   */
  registerAll(processors: Processor[]): void {
    for (const p of processors) {
      this.register(p);
    }
  }

  /**
   * Get a processor by name
   */
  get(name: string): Processor | undefined {
    return this.processors.get(name);
  }

  /**
   * Check if a processor exists
   */
  has(name: string): boolean {
    return this.processors.has(name);
  }

  /**
   * List all registered processor names
   */
  list(): string[] {
    return Array.from(this.processors.keys()).sort();
  }

  /**
   * Get all processors with their descriptions (for LLM discovery)
   */
  describe(): ProcessorDescriptor[] {
    return Array.from(this.processors.values())
      .map(p => ({
        name: p.name,
        description: p.description,
        configSchema: p.configSchema,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Create a processor instance
   */
  createInstance(config: ProcessorConfig, context: ProcessorContext): ProcessorInstance {
    const processor = this.processors.get(config.processor);
    if (!processor) {
      throw new Error(`Unknown processor: ${config.processor}`);
    }
    return processor.create(config, context);
  }

  /**
   * Get the count of registered processors
   */
  get size(): number {
    return this.processors.size;
  }
}

// Singleton instance
let registry: ProcessorRegistry | null = null;

/**
 * Get the global processor registry singleton
 */
export function getProcessorRegistry(): ProcessorRegistry {
  if (!registry) {
    registry = new ProcessorRegistry();
  }
  return registry;
}

/**
 * Reset the registry (primarily for testing)
 */
export function resetProcessorRegistry(): void {
  registry = null;
}
