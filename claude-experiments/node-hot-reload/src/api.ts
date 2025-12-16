/**
 * Hot Reload API
 *
 * These functions are detected by the transformer and handled specially.
 * At runtime (without transform), they just pass through their values.
 */

/**
 * Mark a value to be preserved across hot reloads.
 * Like Clojure's `defonce` - only initializes on first load.
 *
 * @example
 * const cache = defonce(new Map());
 * let counter = defonce(0);
 */
export function defonce<T>(value: T): T {
  return value;
}

/**
 * Mark an expression to only execute once (on first load).
 * Use for side effects like registering handlers.
 *
 * @example
 * once(ipcMain.handle('get', () => getData()));
 * once(setInterval(tick, 1000));
 */
export function once<T>(value: T): T {
  return value;
}
