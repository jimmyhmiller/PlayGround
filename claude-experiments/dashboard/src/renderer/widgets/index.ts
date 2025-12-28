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
  ChartWidget,
  TableWidget,
  StatsWidget,
  TransformWidget,
  LayoutContainer,
  EvalCodeEditor,
  EventDisplay,
  WIDGET_TYPES,
  SELECTORS,
} from './BuiltinWidgets';

// Re-export widget prop types
export type {
  StateValueProps,
  StateListProps,
  DashboardListWidgetProps,
  EvalWidgetProps,
  ChartWidgetProps,
  TableWidgetProps,
  StatsWidgetProps,
  TransformWidgetProps,
  LayoutContainerProps,
  LayoutChildConfig,
  EvalCodeEditorProps,
  EventDisplayProps,
  WidgetTypeConfig,
} from './BuiltinWidgets';
