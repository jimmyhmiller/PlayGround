/**
 * Widgets Module
 *
 * Exports built-in widget types and the createStateWidget factory.
 */

// Factory for creating custom widgets
export { createStateWidget } from './createStateWidget';

// Built-in widget types
export {
  StateValue,
  StateList,
  DashboardList,
  EvalWidget,
  WIDGET_TYPES,
  SELECTORS,
} from './BuiltinWidgets';

// Re-export widget prop types
export type {
  StateValueProps,
  StateListProps,
  DashboardListWidgetProps,
  EvalWidgetProps,
  WidgetTypeConfig,
} from './BuiltinWidgets';
