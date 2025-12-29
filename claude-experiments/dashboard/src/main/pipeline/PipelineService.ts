/**
 * PipelineService
 *
 * Main service for the pipeline system.
 * Handles IPC, initializes processors, and manages the runtime.
 */

import { ipcMain, IpcMainInvokeEvent } from 'electron';
import type { PipelineConfig, ProcessorDescriptor } from '../../types/pipeline';
import { PipelineRuntime } from './PipelineRuntime';
import { getProcessorRegistry } from './ProcessorRegistry';

// Import and register all built-in processors
import { registerCoreProcessors } from './processors/core';
import { registerStatefulProcessors } from './processors/stateful';
import { registerExternalProcessors } from './processors/external';
import { registerDataProcessors } from './processors/data';

/**
 * Event emitter interface
 */
interface EventEmitter {
  emit(type: string, payload: unknown): void;
  subscribe(pattern: string, callback: (event: { type: string; payload: unknown }) => void): () => void;
}

let runtime: PipelineRuntime | null = null;
let initialized = false;

/**
 * Initialize the pipeline service
 */
export function initPipelineService(events: EventEmitter): void {
  if (initialized) {
    console.warn('[PipelineService] Already initialized');
    return;
  }

  // Register all built-in processors
  const registry = getProcessorRegistry();
  registerCoreProcessors(registry);
  registerStatefulProcessors(registry);
  registerExternalProcessors(registry);
  registerDataProcessors(registry);

  console.log(`[PipelineService] Registered ${registry.size} processors`);

  // Create runtime
  runtime = new PipelineRuntime(events);

  initialized = true;
  console.log('[PipelineService] Initialized');
}

/**
 * Get the pipeline runtime
 */
export function getPipelineRuntime(): PipelineRuntime {
  if (!runtime) {
    throw new Error('PipelineService not initialized');
  }
  return runtime;
}

/**
 * Setup IPC handlers for pipeline operations
 */
export function setupPipelineIPC(): void {
  // Start a pipeline
  ipcMain.handle(
    'pipeline:start',
    (_event: IpcMainInvokeEvent, config: PipelineConfig) => {
      return getPipelineRuntime().start(config);
    }
  );

  // Stop a pipeline
  ipcMain.handle(
    'pipeline:stop',
    (_event: IpcMainInvokeEvent, id: string) => {
      return getPipelineRuntime().stop(id);
    }
  );

  // Get pipeline stats
  ipcMain.handle(
    'pipeline:stats',
    (_event: IpcMainInvokeEvent, id: string) => {
      return getPipelineRuntime().getStats(id);
    }
  );

  // Check if pipeline is running
  ipcMain.handle(
    'pipeline:isRunning',
    (_event: IpcMainInvokeEvent, id: string) => {
      return { running: getPipelineRuntime().isRunning(id) };
    }
  );

  // List running pipelines
  ipcMain.handle('pipeline:list', () => {
    return getPipelineRuntime().list();
  });

  // List running pipelines with details
  ipcMain.handle('pipeline:listDetailed', () => {
    return getPipelineRuntime().listDetailed();
  });

  // Stop all pipelines
  ipcMain.handle('pipeline:stopAll', () => {
    getPipelineRuntime().stopAll();
    return { success: true };
  });

  // List available processors
  ipcMain.handle('pipeline:processors', () => {
    return getProcessorRegistry().list();
  });

  // Describe all processors (for LLM)
  ipcMain.handle('pipeline:describeProcessors', (): ProcessorDescriptor[] => {
    return getProcessorRegistry().describe();
  });

  console.log('[PipelineService] IPC handlers registered');
}

/**
 * Cleanup the pipeline service
 */
export function closePipelineService(): void {
  if (runtime) {
    runtime.stopAll();
    runtime = null;
  }
  initialized = false;
  console.log('[PipelineService] Closed');
}
