/**
 * Event Sourcing Utilities
 * ID generation and pattern matching for event types
 */

/**
 * Generate a unique event ID
 * Format: evt_<timestamp>_<random>
 */
function generateEventId() {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `evt_${timestamp}_${random}`;
}

/**
 * Generate a correlation ID for command tracing
 * Format: cmd_<timestamp>_<random>
 */
function generateCorrelationId() {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `cmd_${timestamp}_${random}`;
}

/**
 * Generate a session ID
 * Format: sess_<timestamp>_<random>
 */
function generateSessionId() {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `sess_${timestamp}_${random}`;
}

/**
 * Match an event type against a pattern
 *
 * Patterns:
 *   "*"           - matches all events
 *   "user.login"  - exact match
 *   "user.*"      - matches one level (user.login, user.logout)
 *   "user.**"     - matches all descendants (user.login, user.button.clicked)
 *
 * @param {string} type - The event type to check
 * @param {string} pattern - The pattern to match against
 * @returns {boolean}
 */
function matchesPattern(type, pattern) {
  // Match everything
  if (pattern === '*' || pattern === '**') {
    return true;
  }

  // Deep wildcard: matches the prefix and all descendants
  if (pattern.endsWith('.**')) {
    const prefix = pattern.slice(0, -3);
    return type === prefix || type.startsWith(prefix + '.');
  }

  // Single-level wildcard: matches exactly one segment after prefix
  if (pattern.endsWith('.*')) {
    const prefix = pattern.slice(0, -2);
    if (!type.startsWith(prefix + '.')) {
      return false;
    }
    // Check there's exactly one more segment (no additional dots)
    const remainder = type.slice(prefix.length + 1);
    return !remainder.includes('.');
  }

  // Exact match
  return type === pattern;
}

module.exports = {
  generateEventId,
  generateCorrelationId,
  generateSessionId,
  matchesPattern,
};
