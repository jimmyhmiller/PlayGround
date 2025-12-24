/**
 * Command-Event Pattern Handler
 *
 * Commands are intentions (requests to change state).
 * Events are facts (what actually happened).
 *
 * This helper links commands to their resulting events via correlationId.
 */

import { generateCorrelationId } from './utils';
import type { EventStore } from './EventStore';
import type { DashboardEvent, EventMeta } from '../../types/events';

export interface CommandHandler {
  correlationId: string;
  success<T = unknown>(resultType: string, resultPayload?: T): DashboardEvent<T>;
  failure(error: Error | string): DashboardEvent;
}

export interface CommandHandlerFactory {
  execute(
    commandType: string,
    payload?: unknown,
    meta?: Partial<EventMeta>
  ): CommandHandler;
}

/**
 * Create a command handler bound to an event store
 */
export function createCommandHandler(eventStore: EventStore): CommandHandlerFactory {
  return {
    /**
     * Execute a command
     *
     * @param commandType - The command type (e.g., "counter.increment")
     * @param payload - Command payload
     * @param meta - Additional metadata
     * @returns Handler with correlationId, success(), and failure() methods
     */
    execute(
      commandType: string,
      payload: unknown = {},
      meta: Partial<EventMeta> = {}
    ): CommandHandler {
      const correlationId = generateCorrelationId();

      // Emit command-received event
      eventStore.emit(
        'command.received',
        {
          commandType,
          payload,
        },
        { correlationId, ...meta }
      );

      return {
        correlationId,

        /**
         * Emit a success event
         */
        success<T = unknown>(resultType: string, resultPayload: T = {} as T): DashboardEvent<T> {
          return eventStore.emit(resultType, resultPayload, {
            correlationId,
            ...meta,
          });
        },

        /**
         * Emit a failure event
         */
        failure(error: Error | string): DashboardEvent {
          return eventStore.emit(
            'command.failed',
            {
              commandType,
              error: error instanceof Error ? error.message : String(error),
              stack: error instanceof Error ? error.stack : undefined,
            },
            { correlationId, ...meta }
          );
        },
      };
    },
  };
}
