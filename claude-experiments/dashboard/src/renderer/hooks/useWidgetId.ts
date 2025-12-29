import { createContext, useContext, useId } from 'react';

// ========== Widget ID Context ==========
// Provides stable widget IDs based on config path for state persistence

export interface WidgetIdContextValue {
  /** Base path for this widget (e.g., "pane.routes" or "children.0.children.1") */
  path: string;
  /** Scope ID for namespacing */
  scope: string;
}

export const WidgetIdContext = createContext<WidgetIdContextValue | null>(null);

/**
 * Get a stable widget ID for state persistence.
 * Uses the config path if available, falls back to a generated ID.
 *
 * @param explicitId - Optional explicit ID to use instead of path-based
 * @returns A stable widget ID like "scope::pane.routes" or "scope::children.0"
 */
export function useWidgetId(explicitId?: string): string {
  const ctx = useContext(WidgetIdContext);
  const fallbackId = useId();

  if (explicitId) {
    // Use explicit ID with scope prefix if available
    const id = ctx?.scope ? `${ctx.scope}::${explicitId}` : explicitId;
    console.log(`[useWidgetId] explicitId="${explicitId}", scope="${ctx?.scope}", result="${id}"`);
    return id;
  }

  if (ctx) {
    const id = `${ctx.scope}::${ctx.path}`;
    console.log(`[useWidgetId] path="${ctx.path}", scope="${ctx.scope}", result="${id}"`);
    return id;
  }

  // Fallback for widgets outside WidgetLayout
  console.log(`[useWidgetId] NO CONTEXT, using fallback="${fallbackId}"`);
  return fallbackId;
}
