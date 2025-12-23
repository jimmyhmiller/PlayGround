/**
 * Command-Event Pattern Handler
 *
 * Commands are intentions (requests to change state).
 * Events are facts (what actually happened).
 *
 * This helper links commands to their resulting events via correlationId.
 */

const { generateCorrelationId } = require('./utils');

/**
 * Create a command handler bound to an event store
 *
 * @param {EventStore} eventStore
 * @returns {Object} Command handler with execute method
 */
function createCommandHandler(eventStore) {
  return {
    /**
     * Execute a command
     *
     * @param {string} commandType - The command type (e.g., "counter.increment")
     * @param {Object} payload - Command payload
     * @param {Object} meta - Additional metadata
     * @returns {Object} Handler with correlationId, success(), and failure() methods
     */
    execute(commandType, payload = {}, meta = {}) {
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
         * @param {string} resultType - The resulting event type
         * @param {Object} resultPayload - The result payload
         * @returns {Object} The emitted event
         */
        success(resultType, resultPayload = {}) {
          return eventStore.emit(resultType, resultPayload, {
            correlationId,
            ...meta,
          });
        },

        /**
         * Emit a failure event
         * @param {Error|string} error - The error that occurred
         * @returns {Object} The emitted event
         */
        failure(error) {
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

module.exports = { createCommandHandler };
